"""
Modelo de consulta persistida para AstroData Lab.

Contiene la entidad Consulta que representa la consulta registrada en la base
de datos.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class Consulta(BaseModel):
    """
    Representa una consulta registrada en la base de datos.

    Contiene el registro completo de una consulta realizada por un usuario,
    incluyendo identificadores únicos, timestamp de ejecución y datos del usuario.
    Se genera cuando ConsultaEntrada se procesa y almacena en PostgreSQL.
    """
    id_consulta: int = Field(..., description="Identificador único de la consulta")
    texto_pregunta: str = Field(..., description="Texto de la pregunta o consulta")
    fecha: datetime = Field(..., description="Fecha y hora en que se realizó la consulta")
    id_usuario: int = Field(..., description="Identificador del usuario que realizó la consulta")
