"""
Modelo de luna para AstroData Lab.

Contiene la entidad Luna que representa una luna satélite de un planeta.
"""

from pydantic import BaseModel, Field
from models.domain.base_objeto_astronomico import ObjetoAstronomico


class Luna(ObjetoAstronomico):
    """
    Representa una luna satélite de un planeta.

    Extiende ObjetoAstronomico con referencias al planeta padre y radio ecuatorial.
    Las lunas son cuerpos menores en órbita alrededor de planetas.
    """
    id_planeta: int = Field(..., description="Identificador del planeta al que orbita")
    radio: float = Field(..., description="Radio ecuatorial de la luna en kilómetros")