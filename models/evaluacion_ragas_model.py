"""
Modelo de evaluación RAGAS para AstroData Lab.

Contiene la entidad EvaluacionRAGAS que representa la evaluación completa de
métricas RAGAS de una consulta RAG.
"""

from pydantic import BaseModel, Field
from datetime import datetime


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
