"""
Modelo de resultado para AstroData Lab.

Contiene la entidad Resultado utilizada para representar resultados de
búsqueda y recuperación del sistema RAG.
"""

from pydantic import BaseModel, Field
from typing import Optional


class Resultado(BaseModel):
    """
    Representa un resultado individual de una búsqueda RAG.

    Almacena la puntuación de relevancia y referencias a los documentos/imágenes
    recuperados. Se utiliza internamente para clasificar y presentar resultados
    al usuario según su grado de similitud semántica con la consulta.
    """
    id_resultado: int = Field(..., description="Identificador único del resultado")
    descripcion_resultado: Optional[str] = Field(
        None,
        description="Descripción resumida del resultado para el usuario"
    )
    relevancia: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Puntuación de relevancia semántica (0.0 = irrelevante, 1.0 = máxima relevancia)"
    )
    id_consulta: int = Field(..., description="Referencia a la consulta que generó este resultado")
    id_doc: Optional[int] = Field(
        None,
        description="Referencia opcional al documento recuperado"
    )
    id_imagen: Optional[int] = Field(
        None,
        description="Referencia opcional a la imagen recuperada"
    )