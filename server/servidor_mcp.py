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
from typing import Any, Dict, Callable
from typing import Optional

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


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class ServidorMCPAstroData:
    """
    Servidor MCP principal que expone herramientas de AstroData Lab a Claude.
    """

    def __init__(self) -> None:
        self.servidor = Server("AstroData-Lab")
        self.activo = False

        # Instancias de herramientas (inicializadas después)
        self.tools_rag: ToolsConsultaRAG = None
        self.gestion_objetos: GestionObjetos = None
        self.busqueda_semantica: BusquedaSematica = None
        self.evaluacion_ragas: ToolsEvaluacionRAGAS = None 

        # Mapa nombre_tool -> handler (se llena en _registrar_tools)
        self._handlers: Dict[str, Callable] = {}
        self._tool_defs: list[Tool] = []


    async def _inicializar_base_datos(self) -> None:
        try:
            logger.info("Inicializando pool de conexiones a PostgreSQL...")
            await conexion_bd.iniciar_pool()
            logger.info("✓ Pool de conexiones iniciado (min=5, max=20)")
        except Exception as e:
            logger.error(f"✗ Error al inicializar BD: {e}")
            raise


    async def _inicializar_codificadores(self) -> tuple:
        try:
            logger.info("Inicializando codificadores de embeddings...")

            logger.info(f"  - Cargando modelo de texto: {ajustes.modelo_texto}")
            codificador_texto = CodificadorTexto()
            logger.info("    ✓ Modelo de texto listo (384 dimensiones)")

            logger.info(f"  - Cargando modelo de imagen: {ajustes.modelo_imagen}")
            codificador_imagen = CodificadorImagen()
            logger.info("    ✓ Modelo de imagen listo (512 dimensiones)")

            return codificador_texto, codificador_imagen

        except Exception as e:
            logger.error(f"✗ Error al inicializar codificadores: {e}")
            raise


    async def _inicializar_herramientas(self, codificador_texto, codificador_imagen) -> None:
        try:
            logger.info("Inicializando herramientas MCP...")

            logger.info("  - Instanciando ToolsConsultaRAG (DIP: CodificadorTexto)")
            self.tools_rag = ToolsConsultaRAG(codificador_texto)

            logger.info("  - Instanciando GestionObjetos (DIP: CodificadorTexto)")
            self.gestion_objetos = GestionObjetos(codificador_texto)

            logger.info("  - Instanciando BusquedaSematica (DIP: CodificadorTexto)")
            self.busqueda_semantica = BusquedaSematica(codificador_texto)

            logger.info("  - Instanciando ToolsEvaluacionRAGAS")
            self.evaluacion_ragas = ToolsEvaluacionRAGAS()

            logger.info("✓ Todas las herramientas inicializadas")

        except Exception as e:
            logger.error(f"✗ Error al inicializar herramientas: {e}")
            raise


    def _registrar_tools(self) -> None:
        """
        Recopila todas las definiciones y handlers en listas internas.
        Los decoradores @servidor.list_tools y @servidor.call_tool
        se encargan de exponerlos al SDK.
        """
        try:
            logger.info("Registrando herramientas en servidor MCP...")

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

            # Registrar handlers en el SDK con decoradores
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

            logger.info(f"✓ {len(self._tool_defs)} herramientas registradas exitosamente")

        except Exception as e:
            logger.error(f"✗ Error al registrar herramientas: {e}")
            raise


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
                if nombre_tool == "buscar_documentos_semanticos":
                    resultado = await self.busqueda_semantica.buscar_documentos_semanticos(**argumentos)
                elif nombre_tool == "buscar_imagenes_semanticas":
                    resultado = await self.busqueda_semantica.buscar_imagenes_semanticas(**argumentos)
                elif nombre_tool == "encontrar_planetas_similares":
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


    async def arrancar_servidor(self) -> None:
        logger.info("=" * 70)
        logger.info("🚀 Iniciando servidor MCP de AstroData Lab")
        logger.info("=" * 70)

        try:
            # 1. Inicializar base de datos
            await self._inicializar_base_datos()

            # 2. Inicializar codificadores
            codificador_texto, codificador_imagen = await self._inicializar_codificadores()

            # 3. Inicializar herramientas con inyección de dependencias
            await self._inicializar_herramientas(codificador_texto, codificador_imagen)

            # 4. Registrar herramientas en servidor
            self._registrar_tools()

            # 5. Configurar manejo de señales
            logger.info("Configurando manejadores de señales del sistema...")

            def signal_handler(sig, frame):
                logger.warning(f"Señal {sig} recibida, iniciando cierre...")
                self.activo = False

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            logger.info("✓ Manejadores de señales configurados")

            # 6. Iniciar servidor MCP en modo stdio
            logger.info("=" * 70)
            logger.info("📡 Servidor MCP escuchando en stdio (stdin/stdout)")
            logger.info("   Conectado a Claude Desktop")
            logger.info("=" * 70)

            self.activo = True

            # Inicia transporte stdio (bloquea hasta que se cierre)
            async with stdio_server() as (read_stream, write_stream):
                logger.info("✓ Transporte stdio iniciado y activo")
                await self.servidor.run(
                    read_stream,
                    write_stream,
                    self.servidor.create_initialization_options()
                )

        except Exception as e:
            logger.error(f"✗ Error crítico durante la inicialización: {e}", exc_info=True)
            self.activo = False
            raise

        finally:
            logger.info("Limpiando recursos...")
            await self._cerrar_servidor()


    async def _cerrar_servidor(self) -> None:
        try:
            logger.warning("⏹ Cerrando servidor MCP...")

            if conexion_bd:
                try:
                    logger.info("  - Cerrando pool de conexiones a BD...")
                    await conexion_bd.cerrar_pool()
                    logger.info("    ✓ Pool cerrado")
                except Exception as e:
                    logger.error(f"    ✗ Error al cerrar pool: {e}")

            self.activo = False
            logger.info("✓ Servidor cerrado completamente")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"✗ Error durante cierre: {e}", exc_info=True)


async def main() -> None:
    servidor = ServidorMCPAstroData()
    await servidor.arrancar_servidor()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Servidor interrumpido por usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}", exc_info=True)
        sys.exit(1)