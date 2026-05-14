"""
Modelo de sistema estelar para AstroData Lab.

Contiene la entidad SistemaEstelar que representa sistemas de estrellas
agrupadas gravitacionalmente.
"""

from pydantic import BaseModel, Field
from models.base_objeto_astronomico import ObjetoAstronomico


class SistemaEstelar(ObjetoAstronomico):
    """
    Representa un sistema estelar (grupo de estrellas ligadas gravitacionalmente).

    Extiende ObjetoAstronomico con la referencia a su galaxia contenedora.
    Los sistemas estelares pueden ser binarios, triples, o múltiples.
    """
    id_galaxia: int = Field(..., description="Identificador de la galaxia que contiene al sistema")
