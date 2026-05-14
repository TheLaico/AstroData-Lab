"""
Modelo de característica ambiental para AstroData Lab.

Contiene la entidad CaracteristicaAmbiental que representa medidas
ambientales de un planeta.
"""

from pydantic import BaseModel, Field


class CaracteristicaAmbiental(BaseModel):
    """
    Representa una característica ambiental medible de un planeta.

    Almacena observaciones específicas como presión atmosférica, composición,
    humedad, radiación solar, etc. Cada característica tiene tipo y valor
    para análisis de habitabilidad.
    """
    id_caracteristica: int = Field(..., description="Identificador único de la característica")
    id_planeta: int = Field(..., description="Identificador del planeta observado")
    tipo: str = Field(..., description="Tipo de característica (ej: presión, humedad, radiación)")
    valor: str = Field(..., description="Valor cuantitativo o cualitativo de la característica")