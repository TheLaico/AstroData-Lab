"""
Modelo de imagen para AstroData Lab.

Contiene la entidad Imagen que representa una imagen astronómica registrada
por el sistema.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class Imagen(BaseModel):
    """
    Representa una imagen astronómica registrada en AstroData Lab.

    Almacena la ruta de archivo, metadatos descriptivos y etiquetas de
    clasificación. Cada imagen puede estar asociada a un documento científico
    y cuenta con un embedding CLIP para búsqueda semántica visual.
    """

    id_imagen: int = Field(..., description="Identificador único de la imagen")
    ruta_archivo: str = Field(
        ...,
        min_length=1,
        description="Ruta absoluta o relativa al archivo de imagen"
    )
    descripcion: Optional[str] = Field(
        None,
        description="Descripción textual del contenido visual de la imagen"
    )
    etiquetas: Optional[List[str]] = Field(
        None,
        description="Lista de etiquetas de clasificación (ej: ['galaxia', 'espiral'])"
    )
    id_doc: Optional[int] = Field(
        None,
        description="Referencia opcional al documento científico asociado"
    )
