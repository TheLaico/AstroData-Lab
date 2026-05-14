"""
Modelo de embedding de consulta para AstroData Lab.

Contiene la entidad EmbeddingConsulta que representa el vector de embedding
generado a partir de una consulta.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List


class EmbeddingConsulta(BaseModel):
    """
    Representa el vector embedding generado a partir de una consulta de texto.

    Almacena el embedding numérico de una consulta en pgvector de PostgreSQL.
    Este vector se utiliza para búsqueda semántica similar en la base de datos
    vectorial contra documentos y objetos astronómicos.
    """
    id_embedding: int = Field(..., description="Identificador único del embedding")
    id_consulta: int = Field(..., description="Referencia a la consulta original")
    vector: List[float] = Field(
        ...,
        description="Vector numérico de embedding (típicamente 384 o 768 dimensiones)"
    )
    modelo: str = Field(
        ...,
        description="Nombre del modelo de embedding usado para generar el vector"
    )

    @field_validator('vector')
    @classmethod
    def validar_vector(cls, v: List[float]) -> List[float]:
        """
        Valida que el vector no esté vacío y tenga dimensiones mínimas razonables.

        Args:
            v: Lista de valores del vector a validar

        Returns:
            El vector validado

        Raises:
            ValueError: Si el vector está vacío o tiene menos de 10 dimensiones
        """
        if not v:
            raise ValueError('El vector no puede estar vacío')
        if len(v) < 10:
            raise ValueError('El vector debe tener al menos 10 dimensiones')
        return v
