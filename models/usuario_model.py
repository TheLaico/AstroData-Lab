"""
Modelo de usuario para AstroData Lab.

Contiene la definición de la entidad Usuario que representa al usuario
registrado en el sistema.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class Usuario(BaseModel):
    """
    Representa un usuario registrado en AstroData Lab.

    Almacena información identificativa del usuario: identificador único,
    nombre, correo electrónico y fecha de registro en el sistema.
    """
    id_usuario: int = Field(..., description='Identificador único del usuario')
    nombre: str = Field(..., description='Nombre completo del usuario')
    correo: str = Field(..., description='Dirección de correo electrónico del usuario')
    fecha_registro: Optional[datetime] = Field(
        None,
        description='Fecha y hora en que el usuario se registró en el sistema'
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
