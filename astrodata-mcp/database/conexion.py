"""
Módulo de gestión de conexiones a la base de datos PostgreSQL con pgvector.

Proporciona un pool de conexiones asincrónicas reutilizable para toda la aplicación
AstroData Lab. Utiliza asyncpg para comunicación de alto rendimiento con PostgreSQL
y registra el tipo vector de pgvector para trabajar con embeddings.
"""

import asyncpg
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
from config.ajustes import ajustes


class ConexionBD:
    """
    Gestor de conexiones asincrónicas a la base de datos PostgreSQL.
    
    Mantiene un pool reutilizable de conexiones y proporciona métodos para:
    - Inicializar el pool al arrancar la aplicación
    - Obtener conexiones individuales de forma segura
    - Cerrar el pool al finalizar la aplicación
    
    Diseñada con el Principio Abierto/Cerrado (OCP): puede extenderse con
    funcionalidades como métricas, reintentos o caché sin modificar el código base.
    
    Atributos:
        _pool: Pool de conexiones de asyncpg (None inicialmente)
    """
    
    def __init__(self) -> None:
        """
        Inicializa la instancia del gestor de conexiones.
        
        El pool se crea posteriormente mediante el método iniciar_pool().
        """
        self._pool: Optional[asyncpg.Pool] = None
    
    async def iniciar_pool(self) -> None:
        """
        Crea el pool de conexiones asincrónicas a PostgreSQL.
        
        Lee la URL de conexión desde ajustes.url_base_datos y establece parámetros
        de conexión recomendados. Registra el tipo vector de pgvector para trabajar
        con embeddings numéricos.
        
        Raises:
            ConnectionError: Si falla la conexión a PostgreSQL
            ValueError: Si la URL de base de datos está mal configurada
        
        Raises:
            Exception: Si ocurre un error al crear el pool
        """
        try:
            self._pool = await asyncpg.create_pool(
                ajustes.url_base_datos,
                min_size=5,
                max_size=20,
                command_timeout=60,
                init=self._inicializar_conexion,
            )
        except ValueError as e:
            raise ValueError(
                f"Error en la configuración de la URL de base de datos: {e}"
            ) from e
        except ConnectionError as e:
            raise ConnectionError(
                f"Error al conectar con PostgreSQL en {ajustes.url_base_datos}: {e}"
            ) from e
        except Exception as e:
            raise Exception(
                f"Error inesperado al crear el pool de conexiones: {e}"
            ) from e
    
    async def _inicializar_conexion(self, conexion: asyncpg.Connection) -> None:
        """
        Inicializa cada nueva conexión del pool.
        
        Registra el tipo vector de pgvector para que asyncpg sepa cómo serializar
        y deserializar vectores numéricos. Este método se ejecuta automáticamente
        cuando asyncpg crea nuevas conexiones.
        
        Args:
            conexion: Conexión de asyncpg recién creada
        """
        await conexion.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    async def cerrar_pool(self) -> None:
        """
        Cierra limpiamente el pool de conexiones.
        
        Debe llamarse al finalizar la aplicación para liberar recursos.
        Se ejecuta de forma segura aunque el pool sea None.
        
        Raises:
            Exception: Si ocurre un error al cerrar el pool
        """
        if self._pool is not None:
            try:
                await self._pool.close()
                self._pool = None
            except Exception as e:
                raise Exception(
                    f"Error al cerrar el pool de conexiones: {e}"
                ) from e
    
    @asynccontextmanager
    async def obtener_conexion(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Context manager que proporciona una conexión del pool de forma segura.
        
        Uso:
            async with conexion_bd.obtener_conexion() as conexion:
                resultado = await conexion.fetch("SELECT * FROM tabla")
        
        Yields:
            asyncpg.Connection: Una conexión del pool
        
        Raises:
            RuntimeError: Si el pool no ha sido inicializado
            Exception: Si ocurre un error al obtener una conexión del pool
        """
        if self._pool is None:
            raise RuntimeError(
                "El pool de conexiones no ha sido inicializado. "
                "Llama a iniciar_pool() primero."
            )
        
        try:
            conexion = await self._pool.acquire()
            try:
                yield conexion
            finally:
                await self._pool.release(conexion)
        except Exception as e:
            raise Exception(
                f"Error al obtener conexión del pool: {e}"
            ) from e
    
    def esta_inicializado(self) -> bool:
        """
        Verifica si el pool ha sido inicializado.
        
        Returns:
            True si el pool está disponible, False en caso contrario
        """
        return self._pool is not None


# Instancia global de conexión a base de datos para importar en repositorios
conexion_bd: ConexionBD = ConexionBD()
