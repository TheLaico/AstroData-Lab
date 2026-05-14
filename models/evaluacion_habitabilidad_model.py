"""
Modelo de evaluación de habitabilidad para AstroData Lab.

Contiene la entidad EvaluacionHabitabilidad que representa el potencial
de habitabilidad de un planeta.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class EvaluacionHabitabilidad(BaseModel):
    """
    Representa una evaluación de potencial de habitabilidad de un planeta.

    Realiza un análisis integral del potencial de un planeta para albergar vida
    basándose en factores como temperatura, presión, composición atmosférica,
    radiación solar y otras características ambientales críticas.
    """
    id_eval_habitabilidad: int = Field(..., description="Identificador único de la evaluación")
    id_planeta: int = Field(..., description="Referencia al planeta evaluado")
    puntaje: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Puntaje de habitabilidad normalizado (0.0 = inhabitable, 1.0 = altamente habitable)"
    )
    descripcion: Optional[str] = Field(
        None,
        description="Análisis detallado de factores de habitabilidad"
    )
    fecha: date = Field(..., description="Fecha de la evaluación")
