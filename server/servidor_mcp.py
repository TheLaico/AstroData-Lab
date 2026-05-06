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

# MCP SDK
from mcp.server import Server
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
    
    Gestiona el ciclo de vida completo del servidor MCP:
    - Inicialización de recursos (BD, codificadores)
    - Inyección de dependencias
    - Registro de herramientas
    - Manejo de señales del sistema
    - Cierre limpio de recursos
    
    Implementa ISP (Interface Segregation Principle): solo expone interfaces
    de herramientas públicas a Claude, mantiene repositorios y codificadores
    encapsulados dentro de las tools.
    """
    
    def __init__(self) -> None:
        """
        Inicializa el servidor MCP.
        
        Crea instancia de mcp.Server pero no inicia recursos aún.
        Eso se hace en el método arrancar_servidor().
        """
        self.servidor = Server("AstroData-Lab")
        self.activo = False
        
        # Instancias de herramientas (inicializadas después)
        self.tools_rag: ToolsConsultaRAG = None
        self.gestion_objetos: GestionObjetos = None
        self.busqueda_semantica: BusquedaSematica = None
        self.evaluacion_ragas: ToolsEvaluacionRAGAS = None
    
    
    async def _inicializar_base_datos(self) -> None:
        """
        Inicializa el pool de conexiones a PostgreSQL.
        
        Raises:
            RuntimeError: Si falla la conexión a BD
        """
        try:
            logger.info("Inicializando pool de conexiones a PostgreSQL...")
            await conexion_bd.iniciar_pool()
            logger.info(f"✓ Pool de conexiones iniciado (min=5, max=20)")
        except Exception as e:
            logger.error(f"✗ Error al inicializar BD: {e}")
            raise
    
    
    async def _inicializar_codificadores(self) -> tuple:
        """
        Inicializa codificadores de embeddings.
        
        Carga:
        - CodificadorTexto: all-MiniLM-L6-v2 (384 dimensiones)
        - CodificadorImagen: CLIP ViT-B/32 (512 dimensiones)
        
        Returns:
            Tupla (codificador_texto, codificador_imagen)
        
        Raises:
            RuntimeError: Si falla la carga de modelos
        """
        try:
            logger.info("Inicializando codificadores de embeddings...")
            
            logger.info(f"  - Cargando modelo de texto: {ajustes.modelo_texto}")
            codificador_texto = CodificadorTexto()
            logger.info(f"    ✓ Modelo de texto listo (384 dimensiones)")
            
            logger.info(f"  - Cargando modelo de imagen: {ajustes.modelo_imagen}")
            codificador_imagen = CodificadorImagen()
            logger.info(f"    ✓ Modelo de imagen listo (512 dimensiones)")
            
            return codificador_texto, codificador_imagen
        
        except Exception as e:
            logger.error(f"✗ Error al inicializar codificadores: {e}")
            raise
    
    
    async def _inicializar_herramientas(
        self,
        codificador_texto,
        codificador_imagen
    ) -> None:
        """
        Instancia herramientas MCP con inyección de dependencias.
        
        Aplica DIP: inyecta codificadores sin acoplamiento directo.
        
        Args:
            codificador_texto: Instancia de CodificadorTexto
            codificador_imagen: Instancia de CodificadorImagen
        
        Raises:
            TypeError: Si la inyección de dependencias falla
        """
        try:
            logger.info("Inicializando herramientas MCP...")
            
            # Tools que requieren codificador de texto
            logger.info("  - Instanciando ToolsConsultaRAG (DIP: CodificadorTexto)")
            self.tools_rag = ToolsConsultaRAG(codificador_texto)
            
            logger.info("  - Instanciando GestionObjetos (DIP: CodificadorTexto)")
            self.gestion_objetos = GestionObjetos(codificador_texto)
            
            logger.info("  - Instanciando BusquedaSematica (DIP: CodificadorTexto)")
            self.busqueda_semantica = BusquedaSematica(codificador_texto)
            
            # Tools que no requieren codificador (autosuficientes)
            logger.info("  - Instanciando ToolsEvaluacionRAGAS")
            self.evaluacion_ragas = ToolsEvaluacionRAGAS()
            
            logger.info("✓ Todas las herramientas inicializadas")
        
        except Exception as e:
            logger.error(f"✗ Error al inicializar herramientas: {e}")
            raise
    
    
    def _registrar_tools(self) -> None:
        """
        Registra todas las herramientas en el servidor MCP.
        
        Para cada herramienta:
        1. Obtiene definiciones de Tool
        2. Registra handlers para cada método
        3. Expone solo interfaces públicas (ISP)
        """
        try:
            logger.info("Registrando herramientas en servidor MCP...")
            
            # Herramientas de consulta RAG
            logger.info("  - Registrando tools de ToolsConsultaRAG")
            self._registrar_herramientas_rag()
            
            # Herramientas de gestión de objetos
            logger.info("  - Registrando tools de GestionObjetos")
            self._registrar_herramientas_gestion()
            
            # Herramientas de búsqueda semántica
            logger.info("  - Registrando tools de BusquedaSematica")
            self._registrar_herramientas_busqueda()
            
            # Herramientas de evaluación RAGAS
            logger.info("  - Registrando tools de ToolsEvaluacionRAGAS")
            self._registrar_herramientas_evaluacion()
            
            logger.info("✓ Todas las herramientas registradas exitosamente")
        
        except Exception as e:
            logger.error(f"✗ Error al registrar herramientas: {e}")
            raise
    
    
    def _registrar_herramientas_rag(self) -> None:
        """Registra herramientas de consulta RAG."""
        for tool_def in self.tools_rag.obtener_definiciones_tools():
            handler = self._crear_handler_rag(tool_def.name)
            self.servidor.add_tool(tool_def, handler)
    
    
    def _registrar_herramientas_gestion(self) -> None:
        """Registra herramientas de gestión de objetos."""
        for tool_def in self.gestion_objetos.obtener_definiciones_tools():
            handler = self._crear_handler_gestion(tool_def.name)
            self.servidor.add_tool(tool_def, handler)
    
    
    def _registrar_herramientas_busqueda(self) -> None:
        """Registra herramientas de búsqueda semántica."""
        for tool_def in self.busqueda_semantica.obtener_definiciones_tools():
            handler = self._crear_handler_busqueda(tool_def.name)
            self.servidor.add_tool(tool_def, handler)
    
    
    def _registrar_herramientas_evaluacion(self) -> None:
        """Registra herramientas de evaluación RAGAS."""
        for tool_def in self.evaluacion_ragas.obtener_definiciones_tools():
            handler = self._crear_handler_evaluacion(tool_def.name)
            self.servidor.add_tool(tool_def, handler)
    
    
    def _crear_handler_rag(self, nombre_tool: str) -> Callable:
        """
        Crea handler para herramientas RAG.
        
        Args:
            nombre_tool: Nombre de la herramienta
            
        Returns:
            Función async que ejecuta la herramienta
        """
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
                return TextContent(
                    type="text",
                    text=str({'error': str(e), 'detalles': repr(e)})
                )
        
        return handler
    
    
    def _crear_handler_gestion(self, nombre_tool: str) -> Callable:
        """Crea handler para herramientas de gestión de objetos."""
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
                return TextContent(
                    type="text",
                    text=str({'error': str(e), 'detalles': repr(e)})
                )
        
        return handler
    
    
    def _crear_handler_busqueda(self, nombre_tool: str) -> Callable:
        """Crea handler para herramientas de búsqueda semántica."""
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
                return TextContent(
                    type="text",
                    text=str({'error': str(e), 'detalles': repr(e)})
                )
        
        return handler
    
    
    def _crear_handler_evaluacion(self, nombre_tool: str) -> Callable:
        """Crea handler para herramientas de evaluación RAGAS."""
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
                return TextContent(
                    type="text",
                    text=str({'error': str(e), 'detalles': repr(e)})
                )
        
        return handler
    
    
    async def arrancar_servidor(self) -> None:
        """
        Función principal que inicia el servidor MCP.
        
        Pipeline de inicialización:
        1. Inicializa pool de BD (PostgreSQL + pgvector)
        2. Carga modelos de embeddings (texto e imagen)
        3. Inyecta dependencias en herramientas (DIP)
        4. Registra todas las herramientas en servidor MCP (ISP)
        5. Inicia transporte stdio para comunicación con Claude Desktop
        6. En bloque finally, cierra el pool de BD de forma limpia
        
        Maneja señales del sistema (SIGINT, SIGTERM) para cierre ordenado.
        
        Raises:
            RuntimeError: Si algún paso de inicialización falla
        """
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
            loop = asyncio.get_event_loop()
            
            def signal_handler(sig, frame):
                logger.warning(f"Señal {sig} recibida, iniciando cierre...")
                self.activo = False
                asyncio.create_task(self._cerrar_servidor())
            
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
            async with self.servidor.stdio_transport() as transport:
                logger.info("✓ Transporte stdio iniciado y activo")
                # Este await se mantiene hasta que Claude Desktop cierre la conexión
                await transport.wait()
        
        except Exception as e:
            logger.error(f"✗ Error crítico durante la inicialización: {e}", exc_info=True)
            self.activo = False
            raise
        
        finally:
            logger.info("Limpiando recursos...")
            await self._cerrar_servidor()
    
    
    async def _cerrar_servidor(self) -> None:
        """
        Cierra limpiamente el servidor y libera recursos.
        
        - Cierra el pool de conexiones a BD
        - Registra el cierre en logs
        """
        try:
            logger.warning("⏹ Cerrando servidor MCP...")
            
            # Cerrar pool de conexiones
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
    """
    Punto de entrada del servidor MCP.
    
    Crea instancia de ServidorMCPAstroData y lo inicia.
    """
    servidor = ServidorMCPAstroData()
    await servidor.arrancar_servidor()


if __name__ == "__main__":
    """
    Bloque principal que ejecuta el servidor.
    
    Uso: python servidor_mcp.py
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Servidor interrumpido por usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}", exc_info=True)
        sys.exit(1)
