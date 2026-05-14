"""
Modelo de entrada de consulta para AstroData Lab.

Contiene la entidad ConsultaEntrada utilizada para validar y recibir
nuevas consultas de los usuarios.
"""

from pydantic import BaseModel, Field, field_validator


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