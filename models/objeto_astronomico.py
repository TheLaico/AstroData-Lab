"""
Módulo de modelos de datos para objetos astronómicos en AstroData Lab.

Define la estructura de entidades del dominio astronómico usando Pydantic v2.
Estos modelos representan las tablas del esquema relacional de PostgreSQL
y se utilizan para validación, serialización y comunicación entre capas.
"""

from pydantic import BaseModel, Field
from typing import Optional


# MODELOS DE TIPOS (Entidades de Catálogo)

class TipoGalaxia(BaseModel):
    """
    Representa un tipo de galaxia en la clasificación astronómica.

    Ejemplos: Espiral, Elíptica, Irregular, Lenticular.
    Se utiliza para categorizar galaxias observadas en AstroData Lab.
    """
    id_tipo_galaxia: int = Field(..., description="Identificador único del tipo de galaxia")
    nombre_tipo: str = Field(..., description="Nombre del tipo de galaxia (ej: Espiral, Elíptica)")


class TipoEstrella(BaseModel):
    """
    Representa una clase espectral de estrella en la clasificación de Hertzsprung-Russell.

    Ejemplos: Tipo O, B, A, F, G, K, M (secuencia principal).
    Categoriza las estrellas por temperatura y composición.
    """
    id_tipo_estrella: int = Field(..., description="Identificador único del tipo de estrella")
    nombre_tipo: str = Field(..., description="Clasificación espectral (ej: G2, M5, A0)")


class TipoPlaneta(BaseModel):
    """
    Representa una clasificación de planeta según sus características.

    Ejemplos: Terrestre, Gigante Gaseoso, Enana de Hielo, Supertierra.
    Ayuda a categorizar planetas por tipo y propiedades.
    """
    id_tipo_planeta: int = Field(..., description="Identificador único del tipo de planeta")
    nombre_tipo: str = Field(..., description="Nombre de la clasificación planetaria")





# MODELOS JERÁRQUICOS - OBJETOS ASTRONÓMICOS

class ObjetoAstronomico(BaseModel):
    """
    Clase base que representa cualquier objeto astronómico en AstroData Lab.

    Define atributos comunes a todas las entidades astronómicas:
    identificación, nombre y descripción científica. Sirve como base para
    herencia de galaxias, sistemas, estrellas, planetas y lunas.
    """
    id_objeto: int = Field(..., description="Identificador único del objeto astronómico")
    nombre: str = Field(..., description="Nombre común o designación del objeto")
    descripcion_cientifica: Optional[str] = Field(
        None,
        description="Descripción detallada del objeto con datos científicos"
    )


class Galaxia(ObjetoAstronomico):
    """
    Representa una galaxia en el Universo observable.

    Extiende ObjetoAstronomico con características específicas: tipo de galaxia
    y distancia desde la Tierra. Las galaxias son las estructuras más grandes
    del Universo, conteniendo millones de estrellas.
    """
    id_tipo_galaxia: int = Field(..., description="Referencia al tipo de galaxia")
    distancia: Optional[float] = Field(
        None,
        description="Distancia a la galaxia en años luz"
    )


class SistemaEstelar(ObjetoAstronomico):
    """
    Representa un sistema estelar (grupo de estrellas ligadas gravitacionalmente).

    Extiende ObjetoAstronomico con la referencia a su galaxia contenedora.
    Los sistemas estelares pueden ser binarios, triples, o múltiples.
    """
    id_galaxia: int = Field(..., description="Identificador de la galaxia que contiene al sistema")


class Estrella(ObjetoAstronomico):
    """
    Representa una estrella dentro de un sistema estelar.

    Extiende ObjetoAstronomico con características astrofísicas clave:
    tipo espectral, masa (en masas solares) y temperatura superficial (Kelvin).
    El Sol es la referencia con masa = 1.0 masas solares.
    """
    id_tipo_estrella: int = Field(..., description="Referencia al tipo/clase espectral de estrella")
    id_sistema: int = Field(..., description="Identificador del sistema estelar que contiene la estrella")
    masa: float = Field(..., description="Masa de la estrella en masas solares (1 = masa del Sol)")
    temperatura: int = Field(..., description="Temperatura superficial de la estrella en Kelvin")


class Planeta(ObjetoAstronomico):
    """
    Representa un planeta orbitando una estrella.

    Extiende ObjetoAstronomico con tipo planetario, ubicación orbital (sistema),
    masa (en masas terrestres, donde Tierra = 1.0) y temperatura superficial (Kelvin).
    Los planetas pueden tener lunas satélites.
    """
    id_tipo_planeta: int = Field(..., description="Referencia al tipo de planeta")
    id_sistema: int = Field(..., description="Identificador del sistema estelar al que pertenece")
    masa: float = Field(..., description="Masa del planeta en masas terrestres (1 = masa de la Tierra)")
    temperatura: int = Field(..., description="Temperatura superficial del planeta en Kelvin")


class Luna(ObjetoAstronomico):
    """
    Representa una luna satélite de un planeta.

    Extiende ObjetoAstronomico con referencias al planeta padre y radio ecuatorial.
    Las lunas son cuerpos menores en órbita alrededor de planetas.
    """
    id_planeta: int = Field(..., description="Identificador del planeta al que orbita")
    radio: float = Field(..., description="Radio ecuatorial de la luna en kilómetros")





# MODELO DE CARACTERÍSTICAS AMBIENTALES

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