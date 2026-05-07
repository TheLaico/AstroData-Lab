"""
Módulo de modelos de datos para consultas y usuarios en AstroData Lab.

Define la estructura de entidades relacionadas con consultas al sistema RAG,
usuarios que realizan las consultas y los embeddings generados a partir de ellas.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime


# MODELO DE USUARIO

class Usuario(BaseModel):
    """
    Representa un usuario registrado en AstroData Lab.

    Almacena información identificativa del usuario: identificador único,
    nombre, correo electrónico y fecha de registro en el sistema.
    """
    id_usuario: int = Field(..., description="Identificador único del usuario")
    nombre: str = Field(..., description="Nombre completo del usuario")
    correo: str = Field(..., description="Dirección de correo electrónico del usuario")
    fecha_registro: Optional[datetime] = Field(
        None,
        description="Fecha y hora en que el usuario se registró en el sistema"
    )

    @field_validator('correo')
    @classmethod
    def validar_correo(cls, v: str) -> str:
        """
        Valida que el correo contenga un símbolo @ y un punto.

        Args:
            v: Valor del correo a validar

        Returns:
            El correo validado

        Raises:
            ValueError: Si el formato de correo es inválido
        """
        if '@' not in v or '.' not in v:
            raise ValueError('El correo debe ser válido (contener @ y .)')
        return v


# MODELOS DE CONSULTA

class ConsultaEntrada(BaseModel):
    """
    Modelo para recibir nuevas consultas del usuario.

    Representa los datos mínimos necesarios cuando un usuario realiza una consulta
    al sistema RAG. Se valida antes de procesarse y guardarse en la base de datos.
    """
    texto_pregunta: str = Field(
        ...,
        min_length=3,
        description="Texto de la pregunta o consulta del usuario (mínimo 3 caracteres)"
    )
    id_usuario: int = Field(..., description="Identificador del usuario que realiza la consulta")

    @field_validator('texto_pregunta')
    @classmethod
    def validar_texto_pregunta(cls, v: str) -> str:
        """
        Valida que la pregunta no esté vacía y contenga contenido significativo.

        Args:
            v: Texto de la pregunta a validar

        Returns:
            El texto validado sin espacios en blanco extras

        Raises:
            ValueError: Si el texto es vacío o solo contiene espacios
        """
        texto_limpio = v.strip()
        if not texto_limpio:
            raise ValueError('La pregunta no puede estar vacía')
        return texto_limpio


class Consulta(BaseModel):
    """
    Representa una consulta registrada en la base de datos.

    Contiene el registro completo de una consulta realizada por un usuario,
    incluyendo identificadores únicos, timestamp de ejecución y datos del usuario.
    Se genera cuando ConsultaEntrada se procesa y almacena en PostgreSQL.
    """
    id_consulta: int = Field(..., description="Identificador único de la consulta")
    texto_pregunta: str = Field(..., description="Texto de la pregunta o consulta")
    fecha: datetime = Field(..., description="Fecha y hora en que se realizó la consulta")
    id_usuario: int = Field(..., description="Identificador del usuario que realizó la consulta")


# MODELO DE EMBEDDING DE CONSULTA

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