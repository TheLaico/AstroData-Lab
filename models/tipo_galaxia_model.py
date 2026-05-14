"""
Modelo de tipo de galaxia para AstroData Lab.

Contiene la entidad TipoGalaxia que clasifica tipos de galaxias.
"""

from pydantic import BaseModel, Field


class TipoGalaxia(BaseModel):
    """
    Representa un tipo de galaxia en la clasificación astronómica.

    Ejemplos: Espiral, Elíptica, Irregular, Lenticular.
    Se utiliza para categorizar galaxias observadas en AstroData Lab.
    """
    id_tipo_galaxia: int = Field(..., description="Identificador único del tipo de galaxia")
    nombre_tipo: str = Field(..., description="Nombre del tipo de galaxia (ej: Espiral, Elíptica)")
