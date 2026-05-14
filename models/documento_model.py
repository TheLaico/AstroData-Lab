"""
Modelo de documento para AstroData Lab.

Contiene la entidad Documento que representa un documento científico indexado
por el sistema.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


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
