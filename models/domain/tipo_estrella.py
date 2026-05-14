"""
Modelo de tipo de estrella para AstroData Lab.

Contiene la entidad TipoEstrella que clasifica clases espectrales de estrellas.
"""

from pydantic import BaseModel, Field


class TipoEstrella(BaseModel):
    """
    Representa una clase espectral de estrella en la clasificación de Hertzsprung-Russell.

    Ejemplos: Tipo O, B, A, F, G, K, M (secuencia principal).
    Categoriza las estrellas por temperatura y composición.
    """
    id_tipo_estrella: int = Field(..., description="Identificador único del tipo de estrella")
    nombre_tipo: str = Field(..., description="Clasificación espectral (ej: G2, M5, A0)")