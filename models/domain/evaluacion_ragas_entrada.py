"""
Modelo de entrada de evaluación RAGAS para AstroData Lab.

Contiene la entidad EvaluacionRAGASEntrada utilizada para recibir nuevas
evaluaciones RAGAS del cliente o evaluador.
"""

from pydantic import BaseModel, Field


class EvaluacionRAGASEntrada(BaseModel):
    """
    Modelo para recibir nuevas evaluaciones RAGAS del sistema.

    Versión simplificada de EvaluacionRAGAS sin id_evaluacion ni fecha.
    Se utiliza para recibir evaluaciones del cliente/evaluador y luego se
    almacena en BD con timestamp e ID asignado por la base de datos.
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