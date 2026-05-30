"""
Módulo de gestión de conexiones a la base de datos PostgreSQL con pgvector.
"""

import asyncpg
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
from config.ajustes import ajustes


class ConexionBD:
    """
    Gestor de conexiones asincrónicas a la base de datos PostgreSQL.
    """

    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None

    async def iniciar_pool(self) -> None:
        """
        Crea el pool de conexiones asincrónicas a PostgreSQL.

        Raises:
            ValueError: Si la URL de base de datos está mal configurada
            ConnectionError: Si falla la conexión a PostgreSQL
            Exception: Si ocurre un error inesperado al crear el pool
        """
        try:
            ssl_config = None if ajustes.db_ssl in {"", "false", "none", "disable"} else ajustes.db_ssl
            self._pool = await asyncpg.create_pool(
                ajustes.url_base_datos,
                min_size=2,
                max_size=10,
                command_timeout=60,
                ssl=ssl_config,
                init=self._inicializar_conexion,
            )
        except ValueError as e:
            raise ValueError(
                f"Error en la configuración de la URL de base de datos: {e}"
            ) from e
        except ConnectionError as e:
            raise ConnectionError(
                f"Error al conectar con PostgreSQL: {e}"
            ) from e
        except Exception as e:
            raise Exception(
                f"Error inesperado al crear el pool de conexiones: {e}"
            ) from e

    async def _inicializar_conexion(self, conexion: asyncpg.Connection) -> None:
        """
        Inicializa cada nueva conexión del pool registrando la extensión vector.
        """
        await conexion.execute("CREATE EXTENSION IF NOT EXISTS vector")

    async def cerrar_pool(self) -> None:
        """
        Cierra limpiamente el pool de conexiones.
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
        """
        return self._pool is not None


# Instancia global de conexión a base de datos para importar en repositorios
conexion_bd: ConexionBD = ConexionBD()
