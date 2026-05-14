"""
Modelo de embedding de imagen para AstroData Lab.

Contiene la entidad EmbeddingImagen que representa vectores semánticos de
imágenes astronómicas almacenados en la tabla Embedding_Imagen.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List


class EmbeddingImagen(BaseModel):
    """
    Representa un embedding vectorial generado a partir de una imagen.

    Cada embedding se asocia a una imagen y almacena el nombre del modelo
    usado para generar el vector.
    """
    id_embedding: int = Field(..., description="Identificador único del embedding")
    id_imagen: int = Field(..., description="Referencia a la imagen original")
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