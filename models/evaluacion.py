"""
Módulo de modelos de datos para evaluaciones en AstroData Lab.

Define la estructura de evaluaciones de calidad del sistema RAG usando métricas RAGAS
y evaluaciones de habitabilidad planetaria. Las evaluaciones RAGAS miden la fidelidad,
relevancia y recuperación de contexto del sistema RAG.
"""

from pydantic import BaseModel, Field, computed_field
from typing import Optional, Literal
from datetime import datetime


# MODELOS DE EVALUACIÓN RAGAS

class EvaluacionRAGAS(BaseModel):
    """
    Representa una evaluación completa de métricas RAGAS de una consulta RAG.

    RAGAS (Retrieval-Augmented Generation Assessment) es un framework para evaluar
    sistemas RAG mediante tres métricas principales:
    - Faithfulness (Fidelidad): qué tan factualmente correcta es la respuesta
    - Answer Relevancy (Relevancia): qué tan bien responde a la pregunta
    - Context Recall (Recuperación): qué tan bien el contexto cubre información necesaria

    Se registra en la base de datos con timestamp para rastrear calidad temporal.
    """
    id_evaluacion: int = Field(..., description="Identificador único de la evaluación")
    faithfulness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Métrica de fidelidad: precisión factual de la respuesta (0.0-1.0)"
    )
    answer_relevancy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Métrica de relevancia: qué tan bien responde a la pregunta (0.0-1.0)"
    )
    context_recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Métrica de recuperación: cobertura de información necesaria (0.0-1.0)"
    )
    modelo_eval: str = Field(
        ...,
        description="Nombre del modelo usado para evaluación (ej: 'gpt-3.5-turbo', 'claude-2')"
    )
    fecha: datetime = Field(..., description="Fecha y hora de la evaluación")
    id_consulta: int = Field(..., description="Referencia a la consulta evaluada")


class EvaluacionRAGASEntrada(BaseModel):
    """
    Modelo para recibir nuevas evaluaciones RAGAS del sistema.

    Versión simplificada de EvaluacionRAGAS sin id_evaluacion ni fecha.
    Se utiliza para recibir evaluaciones del cliente/evaluador y luego se
    almacena en BD con timestamp y ID asignado por la base de datos.
    """
    faithfulness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Métrica de fidelidad: precisión factual de la respuesta (0.0-1.0)"
    )
    answer_relevancy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Métrica de relevancia: qué tan bien responde a la pregunta (0.0-1.0)"
    )
    context_recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Métrica de recuperación: cobertura de información necesaria (0.0-1.0)"
    )
    modelo_eval: str = Field(
        ...,
        description="Nombre del modelo usado para evaluación"
    )
    id_consulta: int = Field(..., description="Referencia a la consulta evaluada")


# MODELO DE EVALUACIÓN DE HABITABILIDAD

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
    fecha: datetime = Field(..., description="Fecha y hora de la evaluación")


# MODELO AUXILIAR DE RESUMEN DE EVALUACIÓN

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