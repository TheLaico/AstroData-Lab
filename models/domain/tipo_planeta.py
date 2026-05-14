"""
Modelo de tipo de planeta para AstroData Lab.

Contiene la entidad TipoPlaneta que clasifica categorías planetarias.
"""

from pydantic import BaseModel, Field


class TipoPlaneta(BaseModel):
    """
    Representa una clasificación de planeta según sus características.

    Ejemplos: Terrestre, Gigante Gaseoso, Enana de Hielo, Supertierra.
    Ayuda a categorizar planetas por tipo y propiedades.
    """
    id_tipo_planeta: int = Field(..., description="Identificador único del tipo de planeta")
    nombre_tipo: str = Field(..., description="Nombre de la clasificación planetaria")