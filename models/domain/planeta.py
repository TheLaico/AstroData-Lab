"""
Modelo de planeta para AstroData Lab.

Contiene la entidad Planeta que representa un planeta con información de
masa, temperatura y tipo planetario.
"""

from pydantic import BaseModel, Field
from models.domain.base_objeto_astronomico import ObjetoAstronomico


class Planeta(ObjetoAstronomico):
    """
    Representa un planeta orbitando una estrella.

    Extiende ObjetoAstronomico con tipo planetario, ubicación orbital (sistema),
    masa (en masas terrestres, donde Tierra = 1.0) y temperatura superficial (Kelvin).
    Los planetas pueden tener lunas satélites.
    """
    id_tipo_planeta: int = Field(..., description="Referencia al tipo de planeta")
    id_sistema: int = Field(..., description="Identificador del sistema estelar al que pertenece")
    masa: float = Field(..., description="Masa del planeta en masas terrestres (1 = masa de la Tierra)")
    temperatura: int = Field(..., description="Temperatura superficial del planeta en Kelvin")