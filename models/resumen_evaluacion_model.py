"""
Modelo de resumen de evaluación para AstroData Lab.

Contiene la entidad ResumenEvaluacion que agrega información de evaluación
RAGAS para facilitar la interpretación de resultados.
"""

from pydantic import BaseModel, Field, computed_field
from typing import Literal
from models.evaluacion_ragas_model import EvaluacionRAGAS


class ResumenEvaluacion(BaseModel):
    """
    Resumen analítico de una evaluación RAGAS con métricas agregadas.

    Proporciona un análisis de alto nivel de la evaluación RAGAS, incluyendo
    el promedio de las tres métricas principales y una clasificación de calidad
    derivada que facilita la interpretación rápida por usuarios y sistemas.
    """
    evaluacion: EvaluacionRAGAS = Field(..., description="Evaluación RAGAS completa")

    @computed_field  # type: ignore[misc]
    @property
    def promedio_metricas(self) -> float:
        """
        Calcula el promedio de las tres métricas RAGAS.

        Returns:
            Promedio aritmético de faithfulness, answer_relevancy y context_recall
        """
        return (
            self.evaluacion.faithfulness
            + self.evaluacion.answer_relevancy
            + self.evaluacion.context_recall
        ) / 3.0

    @computed_field  # type: ignore[misc]
    @property
    def calidad(self) -> Literal['baja', 'media', 'alta']:
        """
        Clasifica la calidad general de la evaluación RAGAS.

        Basada en el promedio de las tres métricas:
        - 'baja': promedio < 0.4
        - 'media': promedio entre 0.4 y 0.7
        - 'alta': promedio > 0.7

        Returns:
            Clasificación de calidad ('baja', 'media' o 'alta')
        """
        promedio = self.promedio_metricas

        if promedio < 0.4:
            return 'baja'
        elif promedio <= 0.7:
            return 'media'
        else:
            return 'alta'
