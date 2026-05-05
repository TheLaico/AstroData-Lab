"""
Módulo de interfaz abstracta para codificadores de embeddings en AstroData Lab.

Define el contrato que todos los codificadores de embeddings (texto, imagen) deben cumplir.
Aplica el Principio de Inversión de Dependencias (DIP): las herramientas en tools/ dependen
de esta interfaz abstracta, nunca de implementaciones concretas, permitiendo intercambiar
modelos sin modificar el código que los usa.
"""

from abc import ABC, abstractmethod
from typing import List


class CodificadorBase(ABC):
    """
    Interfaz abstracta para codificadores de embeddings en AstroData Lab.
    
    Define el contrato que deben cumplir todos los codificadores de embeddings,
    ya sean de texto (sentence-transformers) o imagen (CLIP). Esta abstracción
    permite que las herramientas RAG y búsqueda semántica sean agnósticas al
    modelo concreto utilizado.
    
    El Principio de Inversión de Dependencias (DIP) se aplica haciendo que:
    1. Las herramientas en tools/ dependan SOLO de CodificadorBase
    2. Las implementaciones concretas (CodificadorTexto, CodificadorImagen) heredan
       de esta clase sin ser referenciadas directamente por el resto del código
    3. La inyección de dependencias proporciona la instancia correcta en tiempo de ejecución
    
    Ejemplo de uso (desde una herramienta):
        def __init__(self, codificador: CodificadorBase):
            self.codificador = codificador  # No sé ni me importa si es MiniLM o mpnet
        
        async def buscar(self, pregunta: str):
            embedding = await self.codificador.codificar_texto(pregunta)
            # La búsqueda vectorial usa embeddings sin conocer su origen
    """
    
    @abstractmethod
    async def codificar_texto(self, texto: str) -> List[float]:
        """
        Vectoriza un texto individual y retorna su embedding numérico.
        
        Convierte un texto libre (pregunta de usuario, descripción de documento, etc.)
        en un vector denso que representa su significado semántico. Este vector se
        utiliza para búsqueda por similitud coseno contra otros embeddings.
        
        Args:
            texto: Texto a vectorizar. Debe ser no vacío y preferiblemente
                   mayor a 3 caracteres para obtener representaciones significativas.
        
        Returns:
            Lista de flotantes representando el embedding. La dimensión depende del
            modelo concreto (ej: 384 para MiniLM, 768 para mpnet).
        
        Raises:
            ValueError: Si el texto está vacío o es None
            RuntimeError: Si hay error al cargar el modelo o GPU/CPU no disponible
        
        Example:
            >>> codificador = CodificadorTexto()
            >>> embedding = await codificador.codificar_texto("¿Qué es una galaxia?")
            >>> len(embedding)  # Ejemplo: 384 para MiniLM
            384
        """
        pass
    
    @abstractmethod
    async def codificar_textos(self, textos: List[str]) -> List[List[float]]:
        """
        Vectoriza una lista de textos en modo batch para mayor eficiencia.
        
        Procesa múltiples textos simultáneamente (más eficiente que llamar
        codificar_texto() en loop). Se utiliza para vectorizar chunks de documentos,
        múltiples consultas, o llenar tablas de embeddings en la BD.
        
        Args:
            textos: Lista de textos a vectorizar. Puede estar vacía (retorna lista vacía).
                    Cada elemento debe ser no None, aunque puede ser string vacío
                    (aunque no es recomendado semánticamente).
        
        Returns:
            Lista de listas de flotantes. Cada elemento interno es un embedding.
            La longitud del resultado coincide con la longitud de textos.
        
        Raises:
            ValueError: Si algún elemento de textos es None
            RuntimeError: Si hay error al procesar el batch
        
        Example:
            >>> codificador = CodificadorTexto()
            >>> textos = ["Galaxia espiral", "Estrella enana", "Planeta rocoso"]
            >>> embeddings = await codificador.codificar_textos(textos)
            >>> len(embeddings)
            3
            >>> len(embeddings[0])  # Dimensión del embedding
            384
        """
        pass
    
    @abstractmethod
    async def codificar_imagen(self, ruta_imagen: str) -> List[float]:
        """
        Vectoriza una imagen astronómica desde su ruta de archivo y retorna su embedding.
        
        Convierte una imagen (FITS, PNG, JPEG, etc.) en un vector denso que captura
        características visuales relevantes (nebulosas, cúmulos, estructuras galácticas).
        Se utiliza para búsqueda por similitud visual contra otras imágenes.
        
        Args:
            ruta_imagen: Ruta absoluta o relativa al archivo de imagen.
                        Formatos soportados: FITS, PNG, JPEG, TIFF, WebP.
        
        Returns:
            Lista de flotantes representando el embedding visual. La dimensión
            depende del modelo concreto (ej: 512 para CLIP ViT-base).
        
        Raises:
            FileNotFoundError: Si el archivo de imagen no existe
            ValueError: Si el formato de imagen no es soportado
            RuntimeError: Si hay error al cargar o procesar la imagen
        
        Example:
            >>> codificador = CodificadorImagen()
            >>> embedding = await codificador.codificar_imagen("/datos/m31.fits")
            >>> len(embedding)
            512
        """
        pass
    
    @abstractmethod
    async def dimension(self) -> int:
        """
        Retorna la dimensión (número de componentes) del vector embedding.
        
        Información crítica para crear tablas pgvector, validar consistencia de datos,
        y planificar índices de búsqueda vectorial. Diferentes modelos producen
        dimensiones distintas (MiniLM: 384, mpnet: 768, CLIP: 512, etc.).
        
        Returns:
            Número entero positivo representando la dimensión del vector.
        
        Example:
            >>> codificador = CodificadorTexto()  # all-MiniLM-L6-v2
            >>> await codificador.dimension()
            384
            
            >>> codificador_clip = CodificadorImagen()  # clip-vit-base-patch32
            >>> await codificador_clip.dimension()
            512
        """
        pass
    
    @abstractmethod
    async def nombre_modelo(self) -> str:
        """
        Retorna el nombre identitario del modelo de embedding utilizado.
        
        Se registra en la tabla de embeddings para permitir:
        - Rastrear qué modelo generó cada vector (crucial para auditoría)
        - Comparar calidad entre modelos en experimentos
        - Regenerar embeddings si se actualiza el modelo
        - Múltiples versiones de embeddings para el mismo documento
        
        Returns:
            String con el nombre del modelo (ej: 'all-MiniLM-L6-v2', 'openai/clip-vit-base-patch32')
        
        Example:
            >>> codificador = CodificadorTexto()
            >>> await codificador.nombre_modelo()
            'all-MiniLM-L6-v2'
        """
        pass
