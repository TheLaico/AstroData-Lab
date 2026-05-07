"""
Módulo de modelos de datos para resultados en AstroData Lab.

Define la estructura de entidades relacionadas con resultados de búsqueda RAG
y puntuaciones de relevancia. Los modelos de documento e imagen residen en
models.documento.
"""

from pydantic import BaseModel, Field
from typing import Optional
from models.documento import Documento, Imagen



# MODELO DE RESULTADO


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



# MODELO DE RESULTADO DETALLADO


class ResultadoDetallado(Resultado):
    """
    Extiende Resultado con datos completos del documento e imagen asociados.

    Se utiliza para respuestas enriquecidas al usuario, proporcionando no solo
    la puntuación de relevancia sino también el contenido completo del documento
    y los detalles de la imagen para presentación integral en la interfaz.
    """
    documento: Optional[Documento] = Field(
        None,
        description="Objeto Documento completo asociado al resultado"
    )
    imagen: Optional[Imagen] = Field(
        None,
        description="Objeto Imagen completa asociada al resultado"
    )