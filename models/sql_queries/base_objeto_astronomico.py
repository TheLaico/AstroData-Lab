"""
Modelo base de objeto astronómico para AstroData Lab.

Contiene la entidad ObjetoAstronomico que sirve como clase base para las
demás entidades astronómicas.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ObjetoAstronomico(BaseModel):
    """
    Clase base que representa cualquier objeto astronómico en AstroData Lab.

    Define atributos comunes a todas las entidades astronómicas:
    identificación, nombre y descripción científica. Sirve como base para
    herencia de galaxias, sistemas, estrellas, planetas y lunas.
    """
    id_objeto: int = Field(..., description="Identificador único del objeto astronómico")
    nombre: str = Field(..., description="Nombre común o designación del objeto")
    descripcion_cientifica: Optional[str] = Field(
        None,
        description="Descripción detallada del objeto con datos científicos"
    )