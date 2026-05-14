"""
Modelo de resultado detallado para AstroData Lab.

Contiene la entidad ResultadoDetallado que extiende Resultado con detalles
completos de documento e imagen.
"""

from pydantic import BaseModel, Field
from typing import Optional
from models.domain.documento import Documento
from models.domain.imagen import Imagen
from models.domain.resultado import Resultado


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