"""
Modelo de galaxia para AstroData Lab.

Contiene la entidad Galaxia que extiende el objeto astronómico base con datos
específicos de galaxias.
"""

from pydantic import BaseModel, Field
from typing import Optional
from models.base_objeto_astronomico import ObjetoAstronomico


class Galaxia(ObjetoAstronomico):
    """
    Representa una galaxia en el Universo observable.

    Extiende ObjetoAstronomico con características específicas: tipo de galaxia
    y distancia desde la Tierra. Las galaxias son las estructuras más grandes
    del Universo, conteniendo millones de estrellas.
    """
    id_tipo_galaxia: int = Field(..., description="Referencia al tipo de galaxia")
    distancia: Optional[float] = Field(
        None,
        description="Distancia a la galaxia en años luz"
    )
