"""
Modelo de observación astronómica para AstroData Lab.

Contiene la entidad Observacion que representa un registro de observación
capturado por un telescopio sobre un objeto astronómico.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class Observacion(BaseModel):
    """
    Representa una observación realizada por un telescopio.

    Cada observación referencia el telescopio que la realizó y el objeto
    astronómico observado.
    """
    id_observacion: int = Field(..., description="Identificador único de la observación")
    id_telescopio: int = Field(..., description="Referencia al telescopio que realizó la observación")
    id_objeto: int = Field(..., description="Referencia al objeto astronómico observado")
    fecha: date = Field(..., description="Fecha de la observación")
    descripcion: Optional[str] = Field(
        None,
        description="Descripción de la observación o notas adicionales"
    )
