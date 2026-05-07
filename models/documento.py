"""
Módulo de modelos de datos para documentos e imágenes en AstroData Lab.

Define la estructura de entidades relacionadas con documentos científicos
e imágenes astronómicas. Estos modelos son utilizados por el repositorio
de documentos para operaciones CRUD y por los modelos de resultado para
respuestas enriquecidas al usuario.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# MODELO DE DOCUMENTO


class Documento(BaseModel):
    """
    Representa un documento científico indexado en AstroData Lab.

    Almacena los metadatos y contenido textual de publicaciones astronómicas.
    Cada documento puede estar asociado a un objeto astronómico específico y
    se divide en chunks para la generación de embeddings semánticos.
    """

    id_doc: int = Field(..., description="Identificador único del documento")
    titulo: str = Field(
        ...,
        min_length=1,
        description="Título del documento científico"
    )
    idioma: Optional[str] = Field(
        None,
        description="Código de idioma del documento (ej: 'es', 'en')"
    )
    fecha: Optional[datetime] = Field(
        None,
        description="Fecha de publicación o ingesta del documento"
    )
    fuente: Optional[str] = Field(
        None,
        description="Fuente u origen del documento (ej: 'NASA', 'ESA', 'arXiv')"
    )
    contenido_texto: Optional[str] = Field(
        None,
        description="Contenido textual completo del documento"
    )
    id_objeto: Optional[int] = Field(
        None,
        description="Referencia opcional al objeto astronómico asociado"
    )


# MODELO DE IMAGEN


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