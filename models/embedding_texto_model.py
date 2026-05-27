"""
Modelo de embedding de texto para AstroData Lab.

Contiene la entidad EmbeddingTexto que representa vectores semánticos de
chunks de documentos almacenados en la tabla Embedding_Texto.

CAMBIO v2: se agrega campo contenido_chunk para almacenar el texto real
del fragmento junto a su vector, evitando traer contenido_texto completo
del documento en cada búsqueda semántica.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class EmbeddingTexto(BaseModel):
    """
    Representa un embedding vectorial generado a partir de un chunk de texto.

    Cada embedding se asocia a un documento y almacena tanto el vector
    como el texto del fragmento que lo originó. Esto evita hacer JOIN
    con Documento y traer el contenido_texto completo (potencialmente
    cientos de KB) solo para mostrar el fragmento relevante.
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
    contenido_chunk: Optional[str] = Field(
        None,
        description=(
            "Texto real del fragmento que originó este vector. "
            "Permite recuperar el contexto sin traer el documento completo."
        )
    )

    @field_validator('vector')
    @classmethod
    def validar_vector(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError('El vector no puede estar vacío')
        if len(v) < 10:
            raise ValueError('El vector debe tener al menos 10 dimensiones')
        return v