"""
Módulo de configuración y ajustes de la aplicación AstroData Lab.

Gestiona la carga de variables de entorno y proporciona una instancia global
de configuración para toda la aplicación. Este módulo sigue el Principio de
Responsabilidad Única (SRP), enfocándose únicamente en la gestión de configuración.
"""

from typing import Literal
from dotenv import load_dotenv
import os


# Cargar variables de entorno desde el archivo .env
load_dotenv()


class Ajustes:
    """
    Clase que gestiona toda la configuración de la aplicación AstroData Lab.
    
    Carga valores desde variables de entorno con valores por defecto sensatos
    para un sistema RAG híbrido basado en PostgreSQL y pgvector.
    
    Atributos:
        url_base_datos: URL de conexión asincrónica a PostgreSQL usando asyncpg
        modelo_texto: Nombre del modelo de embeddings de texto (sentence transformers)
        modelo_imagen: Nombre del modelo CLIP para embeddings de imagen
        top_k: Número de resultados vectoriales a recuperar por defecto
        estrategia_chunking: Estrategia de división de documentos (fixed, sentence, semantic)
        dimension_vector_texto: Dimensión de los embeddings de texto
        dimension_vector_imagen: Dimensión de los embeddings de imagen
    """
    
    def __init__(self) -> None:
        """
        Inicializa la configuración cargando variables de entorno.
        
        Las variables de entorno tienen prioridad sobre los valores por defecto.
        Si no se proporciona url_base_datos, se genera una URL por defecto
        apuntando a una base de datos PostgreSQL local.
        """
        self.url_base_datos: str = os.getenv(
            'URL_BASE_DATOS',
            'postgresql+asyncpg://usuario:contraseña@localhost:5432/astrodata'
        )
        
        self.modelo_texto: str = os.getenv(
            'MODELO_TEXTO',
            'all-MiniLM-L6-v2'
        )
        
        self.modelo_imagen: str = os.getenv(
            'MODELO_IMAGEN',
            'openai/clip-vit-base-patch32'
        )
        
        self.top_k: int = int(os.getenv('TOP_K', '5'))
        
        self.estrategia_chunking: Literal['fixed', 'sentence', 'semantic'] = os.getenv(
            'ESTRATEGIA_CHUNKING',
            'sentence'
        )  # type: ignore
        
        self.dimension_vector_texto: int = int(os.getenv(
            'DIMENSION_VECTOR_TEXTO',
            '384'
        ))
        
        self.dimension_vector_imagen: int = int(os.getenv(
            'DIMENSION_VECTOR_IMAGEN',
            '512'
        ))
    
    def __repr__(self) -> str:
        """
        Representa la configuración actual en formato legible.
        
        Returns:
            Cadena con los valores de configuración (sin exponer contraseñas).
        """
        return (
            f"Ajustes(\n"
            f"  url_base_datos=****\n"
            f"  modelo_texto='{self.modelo_texto}'\n"
            f"  modelo_imagen='{self.modelo_imagen}'\n"
            f"  top_k={self.top_k}\n"
            f"  estrategia_chunking='{self.estrategia_chunking}'\n"
            f"  dimension_vector_texto={self.dimension_vector_texto}\n"
            f"  dimension_vector_imagen={self.dimension_vector_imagen}\n"
            f")"
        )
    
    def validar_estrategia_chunking(self) -> bool:
        """
        Valida que la estrategia de chunking sea una de las opciones permitidas.
        
        Returns:
            True si la estrategia es válida, False en caso contrario.
        """
        estrategias_validas = {'fixed', 'sentence', 'semantic'}
        return self.estrategia_chunking in estrategias_validas


# Instancia global de configuración para importar en otros módulos
ajustes: Ajustes = Ajustes()
