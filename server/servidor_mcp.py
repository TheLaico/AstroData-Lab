"""
Servidor MCP (Model Context Protocol) principal de AstroData Lab.

Este módulo inicia el servidor MCP que expone las herramientas de AstroData Lab
a Claude Desktop. Orquesta:

1. Inicialización de la base de datos (PostgreSQL + pgvector)
2. Instanciación de codificadores de embeddings (texto e imagen)
3. Inyección de dependencias en herramientas (DIP)
4. Registro de todas las herramientas MCP (ISP)
5. Manejo de cierre limpio con señales del sistema (SIGINT, SIGTERM)

Implementa el patrón de Inversión de Control (IoC) para gestionar el ciclo
de vida de todos los recursos del servidor.

El servidor comunica con Claude Desktop mediante protocolo MCP sobre stdio.
Todos los logs se escriben a stderr para mantener stdout limpio.
"""

import logging
import asyncio
import signal
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Callable, Optional

# Agregar el directorio raíz al path para importar módulos locales
sys.path.insert(0, str(Path(__file__).parent.parent))

# MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configuración y base de datos
from config.ajustes import ajustes
from database.conexion import conexion_bd

# Codificadores (embeddings)
from embeddings.codificador_texto import CodificadorTexto
from embeddings.codificador_imagen import CodificadorImagen

# Herramientas MCP
from tools.consulta_rag import ToolsConsultaRAG
from tools.gestion_objetos import GestionObjetos
from tools.busqueda_semantica import BusquedaSematica
from tools.evaluacion_ragas import ToolsEvaluacionRAGAS


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLA: colores y utilidades de presentación
# ─────────────────────────────────────────────────────────────────────────────

class C:
    """Códigos de color ANSI para la terminal."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    # Texto
    WHITE   = "\033[97m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    GRAY    = "\033[90m"

    # Fondo
    BG_BLUE  = "\033[44m"
    BG_BLACK = "\033[40m"


WIDTH = 68  # Ancho interior del panel (sin bordes)


def _ts() -> str:
    """Timestamp compacto para logs."""
    return datetime.now().strftime("%H:%M:%S")


def _pad(text: str, width: int) -> str:
    """Ajusta texto a ancho fijo, ignorando escapes ANSI para el cálculo."""
    import re
    visible = re.sub(r'\033\[[0-9;]*m', '', text)
    padding = max(0, width - len(visible))
    return text + " " * padding


def banner() -> None:
    """Imprime el banner de inicio del servidor."""
    w = WIDTH + 2  # con los bordes │
    top    = f"╔{'═' * w}╗"
    bottom = f"╚{'═' * w}╝"
    mid    = f"╠{'═' * w}╣"
    empty  = f"║{' ' * w}║"

    title   = "A S T R O D A T A   L A B"
    sub     = "Model Context Protocol Server"
    version = "v1.0.0"

    def center(text: str, color: str = "") -> str:
        import re
        visible = re.sub(r'\033\[[0-9;]*m', '', text)
        pad = (w - len(visible)) // 2
        extra = (w - len(visible)) % 2
        return f"║{' ' * pad}{color}{text}{C.RESET}{' ' * (pad + extra)}║"

    print(f"\n{C.CYAN}{C.BOLD}{top}{C.RESET}", file=sys.stderr)
    print(f"{C.CYAN}{C.BOLD}{empty}{C.RESET}", file=sys.stderr)
    print(center(title, f"{C.BOLD}{C.WHITE}"), file=sys.stderr)
    print(f"{C.CYAN}{C.BOLD}{empty}{C.RESET}", file=sys.stderr)
    print(center(sub, f"{C.CYAN}"), file=sys.stderr)
    print(center(version, f"{C.DIM}{C.GRAY}"), file=sys.stderr)
    print(f"{C.CYAN}{C.BOLD}{empty}{C.RESET}", file=sys.stderr)
    print(f"{C.CYAN}{C.BOLD}{mid}{C.RESET}", file=sys.stderr)

    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    print(center(f"Inicio: {ts}", f"{C.DIM}{C.GRAY}"), file=sys.stderr)
    print(f"{C.CYAN}{C.BOLD}{empty}{C.RESET}", file=sys.stderr)
    print(f"{C.CYAN}{C.BOLD}{bottom}{C.RESET}\n", file=sys.stderr)


def seccion(titulo: str) -> None:
    """Imprime un encabezado de sección."""
    linea = "─" * WIDTH
    print(f"\n{C.BOLD}{C.BLUE}┌{linea}┐{C.RESET}", file=sys.stderr)
    print(f"{C.BOLD}{C.BLUE}│{C.RESET}  {C.BOLD}{C.WHITE}{titulo}{C.RESET}", file=sys.stderr)
    print(f"{C.BOLD}{C.BLUE}└{linea}┘{C.RESET}", file=sys.stderr)


def ok(msg: str, detalle: str = "") -> None:
    det = f"  {C.DIM}{C.GRAY}{detalle}{C.RESET}" if detalle else ""
    print(f"  {C.GREEN}✓{C.RESET}  {msg}{det}", file=sys.stderr)


def info(msg: str, detalle: str = "") -> None:
    det = f"  {C.DIM}{C.GRAY}{detalle}{C.RESET}" if detalle else ""
    print(f"  {C.BLUE}→{C.RESET}  {C.DIM}{_ts()}{C.RESET}  {msg}{det}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"  {C.YELLOW}⚠{C.RESET}  {C.YELLOW}{msg}{C.RESET}", file=sys.stderr)


def err(msg: str) -> None:
    print(f"  {C.RED}✗{C.RESET}  {C.RED}{C.BOLD}{msg}{C.RESET}", file=sys.stderr)


def tabla_tools(tools: list) -> None:
    """Imprime la lista de herramientas registradas en formato tabla."""
    col_w = WIDTH - 4
    print(f"\n  {C.DIM}{'HERRAMIENTA':<40}{'GRUPO':<20}{C.RESET}", file=sys.stderr)
    print(f"  {C.DIM}{'─' * 40}{'─' * 20}{C.RESET}", file=sys.stderr)

    grupos = {
        "rag_query":                    "Consulta RAG",
        "obtener_contexto_objeto":      "Consulta RAG",
        "crear_objeto_astronomico":     "Gestión",
        "obtener_objeto_astronomico":   "Gestión",
        "actualizar_objeto_astronomico":"Gestión",
        "eliminar_objeto_astronomico":  "Gestión",
        "listar_planetas_habitables":   "Gestión",
        "encontrar_planetas_similares": "Búsqueda",
        "evaluar_respuesta_rag":        "Evaluación",
        "obtener_historial_evaluaciones":"Evaluación",
    }

    colores_grupo = {
        "Consulta RAG": C.CYAN,
        "Gestión":      C.BLUE,
        "Búsqueda":     C.MAGENTA,
        "Evaluación":   C.YELLOW,
    }

    for tool in tools:
        grupo = grupos.get(tool.name, "Otro")
        color = colores_grupo.get(grupo, C.WHITE)
        nombre = f"{C.WHITE}{tool.name}{C.RESET}"
        grp    = f"{color}{grupo}{C.RESET}"
        print(f"  {_pad(nombre, 55)}{_pad(grp, 30)}", file=sys.stderr)


def panel_listo(n_tools: int) -> None:
    """Panel final indicando que el servidor está activo."""
    w = WIDTH + 2
    top    = f"╔{'═' * w}╗"
    bottom = f"╚{'═' * w}╝"
    empty  = f"║{' ' * w}║"

    def center(text: str, color: str = "") -> str:
        import re
        visible = re.sub(r'\033\[[0-9;]*m', '', text)
        pad = (w - len(visible)) // 2
        extra = (w - len(visible)) % 2
        return f"║{' ' * pad}{color}{text}{C.RESET}{' ' * (pad + extra)}║"

    print(f"\n{C.GREEN}{C.BOLD}{top}{C.RESET}", file=sys.stderr)
    print(f"{C.GREEN}{C.BOLD}{empty}{C.RESET}", file=sys.stderr)
    print(center("SERVIDOR ACTIVO", f"{C.BOLD}{C.GREEN}"), file=sys.stderr)
    print(f"{C.GREEN}{C.BOLD}{empty}{C.RESET}", file=sys.stderr)
    print(center(f"Escuchando en  stdio  ·  {n_tools} tools registradas", f"{C.DIM}{C.WHITE}"), file=sys.stderr)
    print(center("Conectado a Claude Desktop", f"{C.DIM}{C.GRAY}"), file=sys.stderr)
    print(f"{C.GREEN}{C.BOLD}{empty}{C.RESET}", file=sys.stderr)
    print(f"{C.GREEN}{C.BOLD}{bottom}{C.RESET}\n", file=sys.stderr)


def panel_cierre(duracion: float) -> None:
    """Panel de cierre limpio."""
    mins  = int(duracion // 60)
    segs  = int(duracion % 60)
    uptime = f"{mins}m {segs}s" if mins else f"{segs}s"

    w = WIDTH + 2
    top    = f"╔{'═' * w}╗"
    bottom = f"╚{'═' * w}╝"
    empty  = f"║{' ' * w}║"

    def center(text: str, color: str = "") -> str:
        import re
        visible = re.sub(r'\033\[[0-9;]*m', '', text)
        pad = (w - len(visible)) // 2
        extra = (w - len(visible)) % 2
        return f"║{' ' * pad}{color}{text}{C.RESET}{' ' * (pad + extra)}║"

    print(f"\n{C.GRAY}{top}{C.RESET}", file=sys.stderr)
    print(f"{C.GRAY}{empty}{C.RESET}", file=sys.stderr)
    print(center("SERVIDOR CERRADO", f"{C.BOLD}{C.WHITE}"), file=sys.stderr)
    print(f"{C.GRAY}{empty}{C.RESET}", file=sys.stderr)
    print(center(f"Tiempo activo: {uptime}", f"{C.DIM}{C.GRAY}"), file=sys.stderr)
    print(f"{C.GRAY}{empty}{C.RESET}", file=sys.stderr)
    print(f"{C.GRAY}{bottom}{C.RESET}\n", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING: formato minimalista que aprovecha los helpers de consola
# ─────────────────────────────────────────────────────────────────────────────

class ConsolaHandler(logging.Handler):
    """Handler de logging que usa los helpers de color definidos arriba."""

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        level = record.levelno
        if level >= logging.ERROR:
            err(msg)
        elif level >= logging.WARNING:
            warn(msg)
        else:
            # INFO y DEBUG los omitimos porque ya usamos los helpers directamente
            pass


# Silenciar el logger raíz para evitar duplicados
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(ConsolaHandler())
logger.propagate = False


# ─────────────────────────────────────────────────────────────────────────────
# SERVIDOR
# ─────────────────────────────────────────────────────────────────────────────

class ServidorMCPAstroData:
    """
    Servidor MCP principal que expone herramientas de AstroData Lab a Claude.
    """

    def __init__(self) -> None:
        self.servidor = Server("AstroData-Lab")
        self.activo = False
        self._inicio: float = 0.0

        self.tools_rag: ToolsConsultaRAG         = None
        self.gestion_objetos: GestionObjetos      = None
        self.busqueda_semantica: BusquedaSematica = None
        self.evaluacion_ragas: ToolsEvaluacionRAGAS = None

        self._handlers: Dict[str, Callable] = {}
        self._tool_defs: list[Tool] = []


    async def _inicializar_base_datos(self) -> None:
        seccion("Base de datos")
        info("Conectando a PostgreSQL…")
        try:
            await conexion_bd.iniciar_pool()
            ok("Pool de conexiones iniciado", "min=5  max=20")
        except Exception as e:
            err(f"Error al inicializar BD: {e}")
            raise


    async def _inicializar_codificadores(self) -> tuple:
        seccion("Codificadores de embeddings")
        try:
            info(f"Cargando modelo de texto…", ajustes.modelo_texto)
            codificador_texto = CodificadorTexto()
            ok("Modelo de texto listo", "384 dimensiones")

            info(f"Cargando modelo de imagen…", ajustes.modelo_imagen)
            codificador_imagen = CodificadorImagen()
            ok("Modelo de imagen listo", "512 dimensiones")

            return codificador_texto, codificador_imagen

        except Exception as e:
            err(f"Error al inicializar codificadores: {e}")
            raise


    async def _inicializar_herramientas(self, codificador_texto, codificador_imagen) -> None:
        seccion("Herramientas MCP")
        try:
            info("Instanciando ToolsConsultaRAG")
            self.tools_rag = ToolsConsultaRAG(codificador_texto)
            ok("ToolsConsultaRAG", "DIP: CodificadorTexto")

            info("Instanciando GestionObjetos")
            self.gestion_objetos = GestionObjetos(codificador_texto)
            ok("GestionObjetos", "DIP: CodificadorTexto")

            info("Instanciando BusquedaSematica")
            self.busqueda_semantica = BusquedaSematica(codificador_texto)
            ok("BusquedaSematica", "DIP: CodificadorTexto")

            info("Instanciando ToolsEvaluacionRAGAS")
            self.evaluacion_ragas = ToolsEvaluacionRAGAS()
            ok("ToolsEvaluacionRAGAS")

        except Exception as e:
            err(f"Error al inicializar herramientas: {e}")
            raise


    def _registrar_tools(self) -> None:
        seccion("Registro de tools")
        try:
            grupos = [
                (self.tools_rag.obtener_definiciones_tools(),         self._crear_handler_rag),
                (self.gestion_objetos.obtener_definiciones_tools(),    self._crear_handler_gestion),
                (self.busqueda_semantica.obtener_definiciones_tools(), self._crear_handler_busqueda),
                (self.evaluacion_ragas.obtener_definiciones_tools(),   self._crear_handler_evaluacion),
            ]

            for defs, factory in grupos:
                for tool_def in defs:
                    self._tool_defs.append(tool_def)
                    self._handlers[tool_def.name] = factory(tool_def.name)

            tabla_tools(self._tool_defs)

            @self.servidor.list_tools()
            async def listar_tools() -> list[Tool]:
                return self._tool_defs

            @self.servidor.call_tool()
            async def llamar_tool(name: str, arguments: Dict[str, Any]) -> list[TextContent]:
                handler = self._handlers.get(name)
                if handler is None:
                    return [TextContent(type="text", text=str({'error': f'Herramienta desconocida: {name}'}))]
                result = await handler(arguments or {})
                if isinstance(result, list):
                    return result
                return [result]

            print(f"\n  {C.GREEN}✓{C.RESET}  {C.BOLD}{len(self._tool_defs)} herramientas registradas{C.RESET}", file=sys.stderr)

        except Exception as e:
            err(f"Error al registrar herramientas: {e}")
            raise


    # ── Handlers ─────────────────────────────────────────────────────────────

    def _crear_handler_rag(self, nombre_tool: str) -> Callable:
        async def handler(argumentos: Dict[str, Any]) -> TextContent:
            try:
                if nombre_tool == "rag_query":
                    resultado = await self.tools_rag.rag_query(**argumentos)
                elif nombre_tool == "obtener_contexto_objeto":
                    resultado = await self.tools_rag.obtener_contexto_objeto(**argumentos)
                else:
                    resultado = {'error': f'Herramienta desconocida: {nombre_tool}'}
                return TextContent(type="text", text=str(resultado))
            except Exception as e:
                logger.error(f"Error ejecutando {nombre_tool}: {e}")
                return TextContent(type="text", text=str({'error': str(e), 'detalles': repr(e)}))
        return handler


    def _crear_handler_gestion(self, nombre_tool: str) -> Callable:
        async def handler(argumentos: Dict[str, Any]) -> TextContent:
            try:
                if nombre_tool == "crear_objeto_astronomico":
                    resultado = await self.gestion_objetos.crear_objeto_astronomico(**argumentos)
                elif nombre_tool == "obtener_objeto_astronomico":
                    resultado = await self.gestion_objetos.obtener_objeto_astronomico(**argumentos)
                elif nombre_tool == "actualizar_objeto_astronomico":
                    resultado = await self.gestion_objetos.actualizar_objeto_astronomico(**argumentos)
                elif nombre_tool == "eliminar_objeto_astronomico":
                    resultado = await self.gestion_objetos.eliminar_objeto_astronomico(**argumentos)
                elif nombre_tool == "listar_planetas_habitables":
                    resultado = await self.gestion_objetos.listar_planetas_habitables(**argumentos)
                else:
                    resultado = {'error': f'Herramienta desconocida: {nombre_tool}'}
                return TextContent(type="text", text=str(resultado))
            except Exception as e:
                logger.error(f"Error ejecutando {nombre_tool}: {e}")
                return TextContent(type="text", text=str({'error': str(e), 'detalles': repr(e)}))
        return handler


    def _crear_handler_busqueda(self, nombre_tool: str) -> Callable:
        async def handler(argumentos: Dict[str, Any]) -> TextContent:
            try:
                if nombre_tool == "encontrar_planetas_similares":
                    resultado = await self.busqueda_semantica.encontrar_planetas_similares(**argumentos)
                else:
                    resultado = {'error': f'Herramienta desconocida: {nombre_tool}'}
                return TextContent(type="text", text=str(resultado))
            except Exception as e:
                logger.error(f"Error ejecutando {nombre_tool}: {e}")
                return TextContent(type="text", text=str({'error': str(e), 'detalles': repr(e)}))
        return handler


    def _crear_handler_evaluacion(self, nombre_tool: str) -> Callable:
        async def handler(argumentos: Dict[str, Any]) -> TextContent:
            try:
                if nombre_tool == "evaluar_respuesta_rag":
                    resultado = await self.evaluacion_ragas.evaluar_respuesta_rag(**argumentos)
                elif nombre_tool == "obtener_historial_evaluaciones":
                    resultado = await self.evaluacion_ragas.obtener_historial_evaluaciones(**argumentos)
                else:
                    resultado = {'error': f'Herramienta desconocida: {nombre_tool}'}
                return TextContent(type="text", text=str(resultado))
            except Exception as e:
                logger.error(f"Error ejecutando {nombre_tool}: {e}")
                return TextContent(type="text", text=str({'error': str(e), 'detalles': repr(e)}))
        return handler


    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    async def arrancar_servidor(self) -> None:
        self._inicio = time.monotonic()
        banner()

        try:
            await self._inicializar_base_datos()
            codificador_texto, codificador_imagen = await self._inicializar_codificadores()
            await self._inicializar_herramientas(codificador_texto, codificador_imagen)
            self._registrar_tools()

            seccion("Sistema")
            info("Configurando manejadores de señales…")

            def signal_handler(sig, frame):
                warn(f"Señal {sig} recibida — iniciando cierre limpio…")
                self.activo = False

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            ok("Manejadores SIGINT / SIGTERM configurados")

            self.activo = True
            panel_listo(len(self._tool_defs))

            async with stdio_server() as (read_stream, write_stream):
                await self.servidor.run(
                    read_stream,
                    write_stream,
                    self.servidor.create_initialization_options()
                )

        except Exception as e:
            err(f"Error crítico durante la inicialización: {e}")
            self.activo = False
            raise

        finally:
            await self._cerrar_servidor()


    async def _cerrar_servidor(self) -> None:
        try:
            seccion("Cierre")
            if conexion_bd:
                info("Cerrando pool de conexiones a BD…")
                try:
                    await conexion_bd.cerrar_pool()
                    ok("Pool cerrado")
                except Exception as e:
                    err(f"Error al cerrar pool: {e}")

            self.activo = False
            duracion = time.monotonic() - self._inicio if self._inicio else 0
            panel_cierre(duracion)

        except Exception as e:
            err(f"Error durante cierre: {e}")


# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    servidor = ServidorMCPAstroData()
    await servidor.arrancar_servidor()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # El cierre limpio ya lo maneja _cerrar_servidor
    except Exception as e:
        err(f"Error fatal: {e}")
        sys.exit(1)