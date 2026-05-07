"""
Módulo de modelos de datos para resultados en AstroData Lab.

Define la estructura de entidades relacionadas con resultados de búsqueda RAG,
incluyendo documentos recuperados, imágenes asociadas y puntuaciones de relevancia.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime


# ============================================================================
# MODELO DE DOCUMENTO
# ============================================================================

class Documento(BaseModel):
    """
    Representa un documento en la base de datos de AstroData Lab.
    
    Los documentos son textos científicos, artículos, especificaciones técnicas
    u otro contenido relevante para la exploración astronómica. Se indexan y
    se recuperan mediante búsqueda semántica en consultas RAG.
    """
    id_doc: int = Field(..., description="Identificador único del documento")
    titulo: str = Field(..., description="Título del documento")
    idioma: Optional[str] = Field(
        None,
        description="Código de idioma del documento (ej: 'es', 'en', 'fr')"
    )
    fecha: Optional[datetime] = Field(
        None,
        description="Fecha de publicación o creación del documento"
    )
    fuente: Optional[str] = Field(
        None,
        description="Fuente o referencia del documento (ej: NASA, ESA, artículo académico)"
    )
    contenido_texto: Optional[str] = Field(
        None,
        description="Contenido textual completo o resumen del documento"
    )
    id_objeto: Optional[int] = Field(
        None,
        description="Referencia opcional a un objeto astronómico específico"
    )


# ============================================================================
# MODELO DE IMAGEN
# ============================================================================

class Imagen(BaseModel):
    """
    Representa una imagen astronómica en la base de datos de AstroData Lab.
    
    Las imágenes son fotografías, mapas, diagramas u otros archivos visuales
    relacionados con objetos astronómicos. Pueden estar asociadas a documentos
    y se recuperan mediante búsqueda semántica con embeddings visuales CLIP.
    """
    id_imagen: int = Field(..., description="Identificador único de la imagen")
    ruta_archivo: str = Field(..., description="Ruta del archivo de imagen en el servidor")
    descripcion: Optional[str] = Field(
        None,
        description="Descripción textual del contenido de la imagen"
    )
    etiquetas: Optional[List[str]] = Field(
        None,
        description="Lista de etiquetas descriptivas de la imagen (ej: ['galaxia', 'espiral'])"
    )
    id_doc: Optional[int] = Field(
        None,
        description="Referencia opcional al documento que contiene la imagen"
    )


# ============================================================================
# MODELO DE RESULTADO
# ============================================================================

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
    
    @field_validator('relevancia')
    @classmethod
    def validar_relevancia(cls, v: float) -> float:
        """
        Valida que la relevancia esté normalizada entre 0.0 y 1.0.
        
        Args:
            v: Valor de relevancia a validar
            
        Returns:
            El valor de relevancia validado
            
        Raises:
            ValueError: Si el valor está fuera del rango [0.0, 1.0]
        """
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f'La relevancia debe estar entre 0.0 y 1.0, recibido: {v}'
            )
        return v


# ============================================================================
# MODELO DE RESULTADO DETALLADO
# ============================================================================

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
    
    @field_validator('documento', 'imagen', mode='before')
    @classmethod
    def validar_referencia(cls, v: Optional[dict | Documento | Imagen]) -> Optional[dict | Documento | Imagen]:
        """
        Valida que al menos uno de documento o imagen esté presente.
        
        Args:
            v: Valor del documento o imagen a validar
            
        Returns:
            El valor validado
        """
        return v
