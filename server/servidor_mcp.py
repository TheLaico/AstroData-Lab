"""
Servidor MCP (Model Context Protocol) principal de AstroData Lab.

Este módulo inicia el servidor MCP que expone las herramientas de AstroData Lab
a Claude Desktop. Orquesta:

1. Inicialización de la base de datos (PostgreSQL + pgvector)
2. Instanciación de codificadores de embeddings (texto e imagen)
3. Inyección de dependencias en herramientas (DIP)
4. Registro dinámico de todas las herramientas MCP (OCP)
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
from typing import Any, Dict, Callable, Optional, TYPE_CHECKING

# Agregar el directorio raíz al path para importar módulos locales
sys.path.insert(0, str(Path(__file__).parent.parent))

# MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Prompt, PromptMessage, GetPromptResult

# Configuración y base de datos
from config.ajustes import ajustes
from database.conexion import conexion_bd

if TYPE_CHECKING:
    from database.embeddings.codificador_texto import CodificadorTexto
    from database.embeddings.codificador_imagen import CodificadorImagen

# Herramientas MCP
from tools.consulta_rag import ToolsConsultaRAG
from tools.consulta_hibrida import ToolsConsultaHibrida
from tools.gestion_objetos import GestionObjetos
from tools.busqueda_semantica import BusquedaSematica
from tools.evaluacion_ragas import ToolsEvaluacionRAGAS
from tools.presentacion import ToolsPresentacionAstroData
from tools.modo_profesor import ToolsModoProfesor
from tools.terminal_profesor import ToolsTerminalProfesor


# ─────────────────────────────────────────────────────────────────────────────
# CONTRATO: interfaz que toda clase de tools debe implementar
# ─────────────────────────────────────────────────────────────────────────────

class ToolGroup:
    """
    Clase base que toda clase de herramientas MCP debe extender.

    Contrato:
        - obtener_definiciones_tools() → List[Tool]
            Retorna las definiciones de las tools que gestiona este grupo.
        - ejecutar(nombre_tool, argumentos) → Any
            Despacha la llamada a la herramienta correcta internamente.
            El servidor solo llama a este método; no necesita conocer qué
            tools existen dentro del grupo.

    Al agregar un grupo nuevo basta con:
        1. Extender ToolGroup en la clase de tools.
        2. Añadir la instancia a la lista en _inicializar_herramientas.
    El servidor no requiere ningún otro cambio (OCP).
    """

    def obtener_definiciones_tools(self) -> list[Tool]:
        raise NotImplementedError

    async def ejecutar(self, nombre_tool: str, argumentos: Dict[str, Any]) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} no implementa ejecutar()"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTADORES: envuelven las clases existentes sin modificarlas
# ─────────────────────────────────────────────────────────────────────────────

class GrupoConsultaRAG(ToolGroup):
    """Adaptador que expone ToolsConsultaRAG como ToolGroup."""

    def __init__(self, tools: ToolsConsultaRAG) -> None:
        self._tools = tools

    def obtener_definiciones_tools(self) -> list[Tool]:
        return self._tools.obtener_definiciones_tools()

    async def ejecutar(self, nombre_tool: str, argumentos: Dict[str, Any]) -> Any:
        if nombre_tool == "rag_query":
            return await self._tools.rag_query(**argumentos)
        if nombre_tool == "obtener_contexto_objeto":
            return await self._tools.obtener_contexto_objeto(**argumentos)
        return {'error': f'Herramienta desconocida en GrupoConsultaRAG: {nombre_tool}'}


class GrupoConsultaHibrida(ToolGroup):
    """Adaptador que expone ToolsConsultaHibrida como ToolGroup."""

    def __init__(self, tools: ToolsConsultaHibrida) -> None:
        self._tools = tools

    def obtener_definiciones_tools(self) -> list[Tool]:
        return self._tools.obtener_definiciones_tools()

    async def ejecutar(self, nombre_tool: str, argumentos: Dict[str, Any]) -> Any:
        if nombre_tool == "consulta_hibrida":
            return await self._tools.consulta_hibrida(**argumentos)
        return {'error': f'Herramienta desconocida en GrupoConsultaHibrida: {nombre_tool}'}


class GrupoGestionObjetos(ToolGroup):
    """Adaptador que expone GestionObjetos como ToolGroup."""

    def __init__(self, tools: GestionObjetos) -> None:
        self._tools = tools

    def obtener_definiciones_tools(self) -> list[Tool]:
        return self._tools.obtener_definiciones_tools()

    async def ejecutar(self, nombre_tool: str, argumentos: Dict[str, Any]) -> Any:
        if nombre_tool == "crear_objeto_astronomico":
            return await self._tools.crear_objeto_astronomico(**argumentos)
        if nombre_tool == "obtener_objeto_astronomico":
            return await self._tools.obtener_objeto_astronomico(**argumentos)
        if nombre_tool == "actualizar_objeto_astronomico":
            return await self._tools.actualizar_objeto_astronomico(**argumentos)
        if nombre_tool == "eliminar_objeto_astronomico":
            return await self._tools.eliminar_objeto_astronomico(**argumentos)
        if nombre_tool == "listar_planetas_habitables":
            return await self._tools.listar_planetas_habitables(**argumentos)
        if nombre_tool == "crear_documento_con_embeddings":
            return await self._tools.crear_documento_con_embeddings(**argumentos)
        if nombre_tool == "crear_imagen_con_embedding":
            return await self._tools.crear_imagen_con_embedding(**argumentos)
        if nombre_tool == "generar_embeddings_imagenes_pendientes":
            return await self._tools.generar_embeddings_imagenes_pendientes(**argumentos)
        if nombre_tool == "reemplazar_imagen_con_embedding":
            return await self._tools.reemplazar_imagen_con_embedding(**argumentos)
        if nombre_tool == "eliminar_imagen_astronomica":
            return await self._tools.eliminar_imagen_astronomica(**argumentos)
        if nombre_tool == "crear_telescopio":
            return await self._tools.crear_telescopio(**argumentos)
        if nombre_tool == "obtener_telescopio":
            return await self._tools.obtener_telescopio(**argumentos)
        if nombre_tool == "listar_telescopios":
            return await self._tools.listar_telescopios(**argumentos)
        if nombre_tool == "crear_observacion":
            return await self._tools.crear_observacion(**argumentos)
        if nombre_tool == "listar_observaciones_por_objeto":
            return await self._tools.listar_observaciones_por_objeto(**argumentos)
        if nombre_tool == "listar_observaciones_por_telescopio":
            return await self._tools.listar_observaciones_por_telescopio(**argumentos)
        return {'error': f'Herramienta desconocida en GrupoGestionObjetos: {nombre_tool}'}


class GrupoBusquedaSemantica(ToolGroup):
    """Adaptador que expone BusquedaSematica como ToolGroup."""

    def __init__(self, tools: BusquedaSematica) -> None:
        self._tools = tools

    def obtener_definiciones_tools(self) -> list[Tool]:
        return self._tools.obtener_definiciones_tools()

    async def ejecutar(self, nombre_tool: str, argumentos: Dict[str, Any]) -> Any:
        if nombre_tool == "encontrar_planetas_similares":
            return await self._tools.encontrar_planetas_similares(**argumentos)
        if nombre_tool == "buscar_imagenes_por_descripcion":
            return await self._tools.buscar_imagenes_por_descripcion(**argumentos)
        if nombre_tool == "buscar_imagenes_similares":
            return await self._tools.buscar_imagenes_similares(**argumentos)
        if nombre_tool == "obtener_info_objeto_por_imagen":
            return await self._tools.obtener_info_objeto_por_imagen(**argumentos)
        return {'error': f'Herramienta desconocida en GrupoBusquedaSemantica: {nombre_tool}'}


class GrupoEvaluacionRAGAS(ToolGroup):
    """Adaptador que expone ToolsEvaluacionRAGAS como ToolGroup."""

    def __init__(self, tools: ToolsEvaluacionRAGAS) -> None:
        self._tools = tools

    def obtener_definiciones_tools(self) -> list[Tool]:
        return self._tools.obtener_definiciones_tools()

    async def ejecutar(self, nombre_tool: str, argumentos: Dict[str, Any]) -> Any:
        if nombre_tool == "evaluar_respuesta_rag":
            return await self._tools.evaluar_respuesta_rag(**argumentos)
        if nombre_tool == "obtener_historial_evaluaciones":
            return await self._tools.obtener_historial_evaluaciones(**argumentos)
        return {'error': f'Herramienta desconocida en GrupoEvaluacionRAGAS: {nombre_tool}'}


class GrupoPresentacion(ToolGroup):
    """Adaptador que expone la bienvenida de AstroData Lab como ToolGroup."""

    def __init__(self, tools: ToolsPresentacionAstroData) -> None:
        self._tools = tools

    def obtener_definiciones_tools(self) -> list[Tool]:
        return self._tools.obtener_definiciones_tools()

    async def ejecutar(self, nombre_tool: str, argumentos: Dict[str, Any]) -> Any:
        if nombre_tool == "astro_data_lab":
            return await self._tools.astro_data_lab(**argumentos)
        return {'error': f'Herramienta desconocida en GrupoPresentacion: {nombre_tool}'}


class GrupoModoProfesor(ToolGroup):
    """Adaptador que expone el modo de demostración guiada para el profesor."""

    def __init__(self, tools: ToolsModoProfesor) -> None:
        self._tools = tools

    def obtener_definiciones_tools(self) -> list[Tool]:
        return self._tools.obtener_definiciones_tools()

    async def ejecutar(self, nombre_tool: str, argumentos: Dict[str, Any]) -> Any:
        if nombre_tool == "modo_profesor":
            return await self._tools.modo_profesor(**argumentos)
        return {'error': f'Herramienta desconocida en GrupoModoProfesor: {nombre_tool}'}


class GrupoTerminalProfesor(ToolGroup):
    """Adaptador que expone la terminal de consultas híbridas guiada."""

    def __init__(self, tools: ToolsTerminalProfesor) -> None:
        self._tools = tools

    def obtener_definiciones_tools(self) -> list[Tool]:
        return self._tools.obtener_definiciones_tools()

    async def ejecutar(self, nombre_tool: str, argumentos: Dict[str, Any]) -> Any:
        if nombre_tool == "usarterminal":
            return await self._tools.usarterminal(**argumentos)
        return {'error': f'Herramienta desconocida en GrupoTerminalProfesor: {nombre_tool}'}


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
    w = WIDTH + 2
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
        "astro_data_lab":                "Presentacion",
        "rag_query":                     "Consulta RAG",
        "obtener_contexto_objeto":       "Consulta RAG",
        "crear_objeto_astronomico":      "Gestión",
        "obtener_objeto_astronomico":    "Gestión",
        "actualizar_objeto_astronomico": "Gestión",
        "eliminar_objeto_astronomico":   "Gestión",
        "listar_planetas_habitables":    "Gestión",
        "generar_embeddings_imagenes_pendientes": "Gestión",
        "reemplazar_imagen_con_embedding": "Gestión",
        "eliminar_imagen_astronomica": "Gestión",
        "encontrar_planetas_similares":  "Búsqueda",
        "buscar_imagenes_por_descripcion": "Búsqueda",
        "buscar_imagenes_similares":     "Búsqueda",
        "obtener_info_objeto_por_imagen": "Búsqueda",
        "consulta_hibrida": "Consulta Híbrida",
        "evaluar_respuesta_rag":         "Evaluación",
        "obtener_historial_evaluaciones": "Evaluación",
        "modo_profesor":                 "Demo Profesor",
        "usarterminal":                  "Terminal",
    }

    colores_grupo = {
        "Consulta RAG": C.CYAN,
        "Consulta Híbrida": C.CYAN,
        "Gestión":      C.BLUE,
        "Búsqueda":     C.MAGENTA,
        "Evaluación":    C.YELLOW,
        "Presentacion":  C.GREEN,
        "Demo Profesor": C.GREEN,
        "Terminal":      C.CYAN,
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
# LOGGING
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
        # INFO y DEBUG los omitimos; ya usamos los helpers directamente


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

    El registro de tools es completamente dinámico: _registrar_tools itera
    sobre self._grupos (lista de ToolGroup) y construye el mapa nombre→grupo
    sin ningún if/elif. Para agregar un grupo nuevo solo hay que:
        1. Crear un adaptador que extienda ToolGroup.
        2. Añadir la instancia a self._grupos en _inicializar_herramientas.
    El resto del servidor no cambia.
    """

    def __init__(self) -> None:
        self.servidor = Server("AstroData-Lab")
        self.activo   = False
        self._inicio: float = 0.0

        # Lista de grupos de tools; el servidor itera sobre ella sin conocer
        # el contenido de cada grupo.
        self._grupos: list[ToolGroup] = []

        # Mapa plano nombre_tool → grupo, construido en _registrar_tools
        self._mapa_tools: Dict[str, ToolGroup] = {}
        self._tool_defs:  list[Tool]           = []

    # ── Inicialización ────────────────────────────────────────────────────────

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
            info("Cargando modelo de texto…", ajustes.modelo_texto)
            from database.embeddings.codificador_texto import CodificadorTexto
            from database.embeddings.codificador_imagen import CodificadorImagen

            codificador_texto = CodificadorTexto()
            ok("Modelo de texto listo", "384 dimensiones")

            info("Cargando modelo de imagen…", ajustes.modelo_imagen)
            codificador_imagen = CodificadorImagen()
            ok("Modelo de imagen listo", "512 dimensiones")

            return codificador_texto, codificador_imagen
        except Exception as e:
            err(f"Error al inicializar codificadores: {e}")
            raise

    async def _inicializar_herramientas(
        self,
        codificador_texto: "CodificadorTexto",
        codificador_imagen: "CodificadorImagen"
    ) -> None:
        """
        Instancia los grupos de tools e los añade a self._grupos.

        Para agregar un grupo nuevo en el futuro:
            1. Instanciar la clase de tools.
            2. Envolver en su adaptador ToolGroup.
            3. Añadir con self._grupos.append(...).
        No hay que tocar ningún otro método del servidor.
        """
        seccion("Herramientas MCP")
        try:
            info("Instanciando GrupoPresentacion")
            self._grupos.append(
                GrupoPresentacion(ToolsPresentacionAstroData())
            )
            ok("GrupoPresentacion", "Demo de apertura")

            info("Instanciando GrupoConsultaRAG")
            self._grupos.append(
                GrupoConsultaRAG(ToolsConsultaRAG(codificador_texto))
            )
            ok("GrupoConsultaRAG", "DIP: CodificadorTexto")

            info("Instanciando GrupoConsultaHibrida")
            self._grupos.append(
                GrupoConsultaHibrida(ToolsConsultaHibrida(codificador_texto, codificador_imagen))
            )
            ok("GrupoConsultaHibrida", "DIP: CodificadorTexto + CodificadorImagen")

            info("Instanciando GrupoGestionObjetos")
            self._grupos.append(
                GrupoGestionObjetos(GestionObjetos(codificador_texto, codificador_imagen))
            )
            ok("GrupoGestionObjetos", "DIP: CodificadorTexto + CodificadorImagen")

            info("Instanciando GrupoBusquedaSemantica")
            self._grupos.append(
                GrupoBusquedaSemantica(BusquedaSematica(codificador_texto, codificador_imagen))
            )
            ok("GrupoBusquedaSemantica", "DIP: CodificadorTexto + CodificadorImagen")

            info("Instanciando GrupoEvaluacionRAGAS")
            self._grupos.append(
                GrupoEvaluacionRAGAS(ToolsEvaluacionRAGAS())
            )
            ok("GrupoEvaluacionRAGAS")

            info("Instanciando GrupoModoProfesor")
            self._grupos.append(
                GrupoModoProfesor(ToolsModoProfesor())
            )
            ok("GrupoModoProfesor", "Demo guiada /modotexterprofesor")

            info("Instanciando GrupoTerminalProfesor")
            self._grupos.append(
                GrupoTerminalProfesor(ToolsTerminalProfesor(codificador_texto))
            )
            ok("GrupoTerminalProfesor", "Terminal /usarterminal — consultas híbridas")

        except Exception as e:
            err(f"Error al inicializar herramientas: {e}")
            raise

    # ── Registro dinámico ─────────────────────────────────────────────────────

    def _registrar_tools(self) -> None:
        """
        Registra todas las tools iterando sobre self._grupos.

        Por cada grupo:
            1. Obtiene sus definiciones con obtener_definiciones_tools().
            2. Para cada Tool, guarda grupo → self._mapa_tools[tool.name].
            3. Acumula las definiciones en self._tool_defs.

        El handler único llama grupo.ejecutar(name, arguments) sin ningún
        if/elif; el dispatch interno es responsabilidad de cada ToolGroup.
        """
        seccion("Registro de tools")
        try:
            for grupo in self._grupos:
                for tool_def in grupo.obtener_definiciones_tools():
                    self._tool_defs.append(tool_def)
                    self._mapa_tools[tool_def.name] = grupo

            tabla_tools(self._tool_defs)

            @self.servidor.list_tools()
            async def listar_tools() -> list[Tool]:
                return self._tool_defs

            @self.servidor.call_tool()
            async def llamar_tool(
                name: str,
                arguments: Dict[str, Any]
            ) -> list[TextContent]:
                grupo = self._mapa_tools.get(name)
                if grupo is None:
                    return [TextContent(
                        type="text",
                        text=str({'error': f'Herramienta desconocida: {name}'})
                    )]
                try:
                    resultado = await grupo.ejecutar(name, arguments or {})
                except Exception as e:
                    logger.error(f"Error ejecutando {name}: {e}")
                    resultado = {'error': str(e), 'detalles': repr(e)}

                return [TextContent(type="text", text=str(resultado))]

            # ── Prompts MCP (slash commands en Claude Desktop) ────────────────
            @self.servidor.list_prompts()
            async def listar_prompts() -> list[Prompt]:
                return [
                    Prompt(
                        name="modotexterprofesor",
                        description=(
                            "Activa la demostración guiada de AstroData Lab para el "
                            "Profesor Paolo. Muestra saludo personalizado, estado en vivo "
                            "del sistema, menú de opciones y activa el modo paso a paso."
                        ),
                    ),
                    Prompt(
                        name="usarterminal",
                        description=(
                            "Terminal de consultas híbridas. Escribe el intent como "
                            "comentario SQL (-- texto) y la tool ejecuta la consulta "
                            "híbrida real mostrando todos los pasos del pipeline."
                        ),
                        arguments=[
                            {
                                "name": "entrada",
                                "description": (
                                    "Bloque con comentarios SQL (--) que describen el intent. "
                                    "El código SQL después del comentario es ignorado."
                                ),
                                "required": True,
                            }
                        ],
                    ),
                ]

            @self.servidor.get_prompt()
            async def obtener_prompt(name: str, arguments: dict | None = None) -> GetPromptResult:
                if name == "modotexterprofesor":
                    return GetPromptResult(
                        messages=[
                            PromptMessage(
                                role="user",
                                content=TextContent(
                                    type="text",
                                    text=(
                                        "Invoca ahora mismo la tool modo_profesor "
                                        "con nombre_profesor='Paolo'. "
                                        "Cuando recibas el resultado, toma el campo "
                                        "'markdown_final' y muéstralo EXACTAMENTE "
                                        "como está, sin agregar ningún texto propio "
                                        "antes ni después. Solo el markdown renderizado."
                                    ),
                                ),
                            )
                        ]
                    )

                if name == "usarterminal":
                    entrada = (arguments or {}).get("entrada", "")
                    if not entrada:
                        entrada = (
                            "-- escribe aquí tu intent\n"
                            "-- ejemplo: planetas similares a la Tierra con idioma=es\n"
                        )
                    return GetPromptResult(
                        messages=[
                            PromptMessage(
                                role="user",
                                content=TextContent(
                                    type="text",
                                    text=(
                                        f"Invoca la tool usarterminal con el siguiente "
                                        f"argumento entrada:\n\n{entrada}\n\n"
                                        "Cuando recibas el resultado, renderiza el campo "
                                        "presentacion_markdown completo con todo el formato "
                                        "Markdown. Muestra las tablas y bloques de código "
                                        "exactamente como están."
                                    ),
                                ),
                            )
                        ]
                    )

                return GetPromptResult(messages=[])

            ok("Prompt /modotexterprofesor registrado", "slash command Claude Desktop")
            ok("Prompt /usarterminal registrado",       "terminal consultas híbridas")

            print(
                f"\n  {C.GREEN}✓{C.RESET}  "
                f"{C.BOLD}{len(self._tool_defs)} herramientas registradas{C.RESET}",
                file=sys.stderr
            )

        except Exception as e:
            err(f"Error al registrar herramientas: {e}")
            raise

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
