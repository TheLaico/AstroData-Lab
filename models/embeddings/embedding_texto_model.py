"""
Modelo de embedding de texto para AstroData Lab.

Contiene la entidad EmbeddingTexto que representa vectores semánticos de
chunks de documentos almacenados en la tabla Embedding_Texto.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List


class EmbeddingTexto(BaseModel):
    """
    Representa un embedding vectorial generado a partir de un chunk de texto.

    Cada embedding se asocia a un documento y se etiqueta con la estrategia de
    chunking utilizada para su generación.
    """
    id_embedding: int = Field(..., description="Identificador único del embedding")
    id_doc: int = Field(..., description="Referencia al documento fuente")
    chunk_id: int = Field(..., description="Identificador del chunk dentro del documento")
    estrategia_chunking: str = Field(
        ...,
        description="Estrategia de chunking usada para generar este fragmento"
    )
    vector: List[float] = Field(..., description="Vector de embedding numérico")
    modelo: str = Field(..., description="Modelo que generó el embedding")

    @field_validator('vector')
    @classmethod
    def validar_vector(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError('El vector no puede estar vacío')
        if len(v) < 10:
            raise ValueError('El vector debe tener al menos 10 dimensiones')
        return v