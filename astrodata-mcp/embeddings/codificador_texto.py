"""
Módulo de codificador de embeddings de texto para AstroData Lab.

Implementa la interfaz CodificadorBase usando la librería sentence-transformers,
que proporciona modelos preentrenados eficientes para generar embeddings semánticos
de textos en lenguaje natural. Se utiliza para vectorizar consultas, documentos y
descripciones científicas.
"""

from sentence_transformers import SentenceTransformer
from typing import List
from config.ajustes import ajustes
from embeddings.interfaz_codificador import CodificadorBase


class CodificadorTexto(CodificadorBase):
    """
    Implementación concreta de CodificadorBase para embeddings de texto.
    
    Utiliza sentence-transformers para generar vectores densos que capturan
    significado semántico de textos. El modelo se carga una sola vez al instanciar
    y se reutiliza en todas las codificaciones posteriores para eficiencia.
    
    El modelo utilizado se obtiene desde ajustes.modelo_texto, permitiendo cambiar
    el modelo sin modificar el código (ej: cambiar de all-MiniLM-L6-v2 a all-mpnet-base-v2).
    
    Atributos:
        _modelo: Instancia cargada del modelo SentenceTransformer
        _nombre_modelo: Identificador del modelo para registro en BD
        _dimension: Dimensión del vector embedding (típicamente 384 para MiniLM)
    """
    
    def __init__(self) -> None:
        """
        Inicializa el codificador cargando el modelo de sentence-transformers.
        
        Lee el nombre del modelo y dimensión desde ajustes. Descarga el modelo
        de HuggingFace la primera vez (cachea en ~/.cache/huggingface).
        
        Raises:
            RuntimeError: Si falla la descarga o carga del modelo
            ValueError: Si el nombre del modelo no existe en HuggingFace
        """
        try:
            self._nombre_modelo = ajustes.modelo_texto
            self._modelo = SentenceTransformer(self._nombre_modelo)
            self._dimension = ajustes.dimension_vector_texto
        except Exception as e:
            raise RuntimeError(
                f"Error al cargar el modelo de embeddings '{self._nombre_modelo}': {e}"
            ) from e
    
    async def codificar_texto(self, texto: str) -> List[float]:
        """
        Codifica un texto individual en un vector embedding de 384 dimensiones.
        
        Normaliza el texto (elimina espacios extras, convierte a minúsculas) antes
        de codificar para asegurar consistencia. Utiliza la GPU si está disponible.
        
        Args:
            texto: Texto a vectorizar. Se normaliza automáticamente.
        
        Returns:
            Lista de 384 flotantes representando el embedding semántico.
        
        Raises:
            ValueError: Si el texto es None o vacío
            RuntimeError: Si hay error en la codificación
        
        Example:
            >>> codificador = CodificadorTexto()
            >>> embedding = await codificador.codificar_texto("Galaxia espiral")
            >>> len(embedding)
            384
        """
        if not texto or not isinstance(texto, str):
            raise ValueError("El texto debe ser una cadena no vacía")
        
        try:
            # Normalizar el texto
            texto_normalizado = texto.strip().lower()
            
            if not texto_normalizado:
                raise ValueError("El texto después de normalización está vacío")
            
            # Codificar usando sentence-transformers
            embedding = self._modelo.encode(
                texto_normalizado,
                convert_to_tensor=False
            )
            
            # Convertir a List[float]
            return embedding.tolist()
        
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Error al codificar texto: {e}"
            ) from e
    
    async def codificar_textos(self, textos: List[str]) -> List[List[float]]:
        """
        Codifica una lista de textos en modo batch para máxima eficiencia.
        
        Procesa todos los textos simultáneamente aprovechando la vectorización
        de NumPy y aceleración GPU. Es significativamente más rápido que llamar
        codificar_texto() en un loop.
        
        Args:
            textos: Lista de textos a codificar. Puede estar vacía.
                    Cada elemento se normaliza igual que en codificar_texto().
        
        Returns:
            Lista de listas de flotantes. Mantiene orden con entrada.
            Ejemplo: [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
        
        Raises:
            ValueError: Si algún elemento no es string o está vacío después de normalización
            RuntimeError: Si hay error en la codificación batch
        
        Example:
            >>> codificador = CodificadorTexto()
            >>> textos = ["Galaxia espiral", "Estrella binaria", "Planeta oceánico"]
            >>> embeddings = await codificador.codificar_textos(textos)
            >>> len(embeddings)
            3
            >>> len(embeddings[0])
            384
        """
        if not textos:
            return []
        
        try:
            # Validar y normalizar todos los textos
            textos_normalizados = []
            for texto in textos:
                if not isinstance(texto, str):
                    raise ValueError(
                        f"Todos los elementos deben ser strings, recibido: {type(texto)}"
                    )
                texto_norm = texto.strip().lower()
                if not texto_norm:
                    raise ValueError(
                        "Ningún texto puede estar vacío después de normalización"
                    )
                textos_normalizados.append(texto_norm)
            
            # Codificar en batch
            embeddings = self._modelo.encode(
                textos_normalizados,
                convert_to_tensor=False,
                batch_size=32,  # Procesar 32 textos por paso
                show_progress_bar=False
            )
            
            # Convertir a List[List[float]]
            return embeddings.tolist()
        
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Error al codificar batch de textos: {e}"
            ) from e
    
    async def codificar_imagen(self, ruta_imagen: str) -> List[float]:
        """
        No implementado para este codificador.
        
        CodificadorTexto está diseñado exclusivamente para textos.
        Para imágenes, usa CodificadorImagen que implementa CLIP.
        
        Args:
            ruta_imagen: Ignorado
        
        Raises:
            NotImplementedError: Siempre, ya que este codificador no maneja imágenes
        """
        raise NotImplementedError(
            "CodificadorTexto no puede codificar imágenes. "
            "Usa CodificadorImagen con el modelo CLIP para imágenes."
        )
    
    async def dimension(self) -> int:
        """
        Retorna la dimensión del vector embedding de texto.
        
        Returns:
            384 para all-MiniLM-L6-v2, 768 para all-mpnet-base-v2, etc.
            El valor específico se obtiene desde ajustes.dimension_vector_texto.
        
        Example:
            >>> codificador = CodificadorTexto()
            >>> await codificador.dimension()
            384
        """
        return self._dimension
    
    async def nombre_modelo(self) -> str:
        """
        Retorna el nombre del modelo de embeddings de texto.
        
        Se utiliza para:
        - Registrar en tabla Embedding_Texto qué modelo generó cada vector
        - Permitir múltiples versiones de embeddings para el mismo documento
        - Comparar calidad entre modelos diferentes
        
        Returns:
            Nombre del modelo (ej: 'all-MiniLM-L6-v2')
        
        Example:
            >>> codificador = CodificadorTexto()
            >>> await codificador.nombre_modelo()
            'all-MiniLM-L6-v2'
        """
        return self._nombre_modelo
