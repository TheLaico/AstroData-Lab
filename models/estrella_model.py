"""
Modelo de estrella para AstroData Lab.

Contiene la entidad Estrella que representa una estrella con propiedades
astrofísicas y relación al sistema estelar.
"""

from pydantic import BaseModel, Field
from models.base_objeto_astronomico import ObjetoAstronomico


class Estrella(ObjetoAstronomico):
    """
    Representa una estrella dentro de un sistema estelar.

    Extiende ObjetoAstronomico con características astrofísicas clave:
    tipo espectral, masa (en masas solares) y temperatura superficial (Kelvin).
    El Sol es la referencia con masa = 1.0 masas solares.
    """
    id_tipo_estrella: int = Field(..., description="Referencia al tipo/clase espectral de estrella")
    id_sistema: int = Field(..., description="Identificador del sistema estelar que contiene la estrella")
    masa: float = Field(..., description="Masa de la estrella en masas solares (1 = masa del Sol)")
    temperatura: int = Field(..., description="Temperatura superficial de la estrella en Kelvin")
