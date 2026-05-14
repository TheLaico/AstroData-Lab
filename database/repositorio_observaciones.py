"""
Repositorio de telescopios y observaciones para AstroData Lab.

Proporciona la capa de acceso a datos para las tablas Telescopio y Observacion.
"""

from typing import List, Optional
from datetime import date
from database.conexion import conexion_bd
from models.telescopio_model import Telescopio
from models.observacion_model import Observacion


class RepositorioObservaciones:
    """
    Repositorio para gestionar telescopios y observaciones astronómicas.

    Encapsula operaciones CRUD sobre las tablas Telescopio y Observacion.
    """

    async def crear_telescopio(
        self,
        nombre: str,
        tipo: Optional[str] = None,
        ubicacion: Optional[str] = None
    ) -> Telescopio:
        if not nombre or not nombre.strip():
            raise ValueError("El nombre del telescopio no puede estar vacío")

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_telescopio = await conexion.fetchval(
                    """
                    INSERT INTO Telescopio (nombre, tipo, ubicacion)
                    VALUES ($1, $2, $3)
                    RETURNING id_telescopio
                    """,
                    nombre.strip(),
                    tipo,
                    ubicacion
                )

                return Telescopio(
                    id_telescopio=id_telescopio,
                    nombre=nombre.strip(),
                    tipo=tipo,
                    ubicacion=ubicacion
                )

        except Exception as e:
            raise RuntimeError(
                f"Error al crear telescopio '{nombre}': {e}"
            ) from e

    async def obtener_telescopio_por_id(self, id_telescopio: int) -> Optional[Telescopio]:
        if not isinstance(id_telescopio, int) or id_telescopio <= 0:
            raise ValueError("id_telescopio debe ser un entero positivo")

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                fila = await conexion.fetchrow(
                    """
                    SELECT id_telescopio, nombre, tipo, ubicacion
                    FROM Telescopio
                    WHERE id_telescopio = $1
                    """,
                    id_telescopio
                )

                if not fila:
                    return None

                return Telescopio(
                    id_telescopio=fila['id_telescopio'],
                    nombre=fila['nombre'],
                    tipo=fila['tipo'],
                    ubicacion=fila['ubicacion']
                )

        except Exception as e:
            raise RuntimeError(
                f"Error al obtener telescopio {id_telescopio}: {e}"
            ) from e

    async def listar_telescopios(self) -> List[Telescopio]:
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                filas = await conexion.fetch(
                    """
                    SELECT id_telescopio, nombre, tipo, ubicacion
                    FROM Telescopio
                    ORDER BY nombre
                    """
                )

                return [
                    Telescopio(
                        id_telescopio=fila['id_telescopio'],
                        nombre=fila['nombre'],
                        tipo=fila['tipo'],
                        ubicacion=fila['ubicacion']
                    )
                    for fila in filas
                ]

        except Exception as e:
            raise RuntimeError(
                f"Error al listar telescopios: {e}"
            ) from e

    async def crear_observacion(
        self,
        id_telescopio: int,
        id_objeto: int,
        fecha: date,
        descripcion: Optional[str] = None
    ) -> Observacion:
        if id_telescopio <= 0:
            raise ValueError("id_telescopio debe ser positivo")
        if id_objeto <= 0:
            raise ValueError("id_objeto debe ser positivo")
        if not fecha:
            raise ValueError("La fecha de la observación no puede ser vacía")

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_observacion = await conexion.fetchval(
                    """
                    INSERT INTO Observacion (id_telescopio, id_objeto, fecha, descripcion)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id_observacion
                    """,
                    id_telescopio,
                    id_objeto,
                    fecha,
                    descripcion
                )

                return Observacion(
                    id_observacion=id_observacion,
                    id_telescopio=id_telescopio,
                    id_objeto=id_objeto,
                    fecha=fecha,
                    descripcion=descripcion
                )

        except Exception as e:
            raise RuntimeError(
                f"Error al crear observación: {e}"
            ) from e

    async def listar_observaciones_por_objeto(self, id_objeto: int) -> List[Observacion]:
        if id_objeto <= 0:
            raise ValueError("id_objeto debe ser positivo")

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                filas = await conexion.fetch(
                    """
                    SELECT id_observacion, id_telescopio, id_objeto, fecha, descripcion
                    FROM Observacion
                    WHERE id_objeto = $1
                    ORDER BY fecha DESC
                    """,
                    id_objeto
                )

                return [
                    Observacion(
                        id_observacion=fila['id_observacion'],
                        id_telescopio=fila['id_telescopio'],
                        id_objeto=fila['id_objeto'],
                        fecha=fila['fecha'],
                        descripcion=fila['descripcion']
                    )
                    for fila in filas
                ]

        except Exception as e:
            raise RuntimeError(
                f"Error al listar observaciones del objeto {id_objeto}: {e}"
            ) from e

    async def listar_observaciones_por_telescopio(self, id_telescopio: int) -> List[Observacion]:
        if id_telescopio <= 0:
            raise ValueError("id_telescopio debe ser positivo")

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                filas = await conexion.fetch(
                    """
                    SELECT id_observacion, id_telescopio, id_objeto, fecha, descripcion
                    FROM Observacion
                    WHERE id_telescopio = $1
                    ORDER BY fecha DESC
                    """,
                    id_telescopio
                )

                return [
                    Observacion(
                        id_observacion=fila['id_observacion'],
                        id_telescopio=fila['id_telescopio'],
                        id_objeto=fila['id_objeto'],
                        fecha=fila['fecha'],
                        descripcion=fila['descripcion']
                    )
                    for fila in filas
                ]

        except Exception as e:
            raise RuntimeError(
                f"Error al listar observaciones del telescopio {id_telescopio}: {e}"
            ) from e
