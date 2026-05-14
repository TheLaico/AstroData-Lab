"""
Modelo de telescopio para AstroData Lab.

Contiene la entidad Telescopio que representa telescopios y antenas
astronómicas registradas en el sistema.
"""

from pydantic import BaseModel, Field
from typing import Optional


class Telescopio(BaseModel):
    """
    Representa un telescopio o instrumento de observación en AstroData Lab.

    Se usa para asociar observaciones a una instalación específica.
    """
    id_telescopio: int = Field(..., description="Identificador único del telescopio")
    nombre: str = Field(..., description="Nombre del telescopio o instalación")
    tipo: Optional[str] = Field(
        None,
        description="Tipo de telescopio (ej: 'Óptico', 'Radio', 'Espacial')"
    )
    ubicacion: Optional[str] = Field(
        None,
        description="Ubicación geográfica o plataforma del telescopio"
    )