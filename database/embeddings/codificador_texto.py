"""
Módulo de codificador de embeddings de texto para AstroData Lab.

Implementa la interfaz CodificadorBase usando la librería sentence-transformers,
que proporciona modelos preentrenados eficientes para generar embeddings semánticos
de textos en lenguaje natural. Se utiliza para vectorizar consultas, documentos y
descripciones científicas.

CAMBIO v2: codificar_texto y codificar_textos usan run_in_executor para no
bloquear el event loop de asyncio durante la inferencia síncrona de PyTorch.
"""

import asyncio
from functools import partial
from sentence_transformers import SentenceTransformer
from typing import List
from config.ajustes import ajustes
from database.embeddings.interfaz_codificador import CodificadorBase


class CodificadorTexto(CodificadorBase):
    """
    Implementación concreta de CodificadorBase para embeddings de texto.

    Utiliza sentence-transformers para generar vectores densos que capturan
    significado semántico de textos. El modelo se carga una sola vez al instanciar
    y se reutiliza en todas las codificaciones posteriores para eficiencia.

    IMPORTANTE: sentence-transformers corre sobre PyTorch síncrono. Llamarlo
    directamente dentro de un método async bloquea el event loop completo,
    congelando el servidor MCP mientras dura la inferencia. Por eso todos los
    métodos de codificación delegan al executor por defecto (ThreadPoolExecutor)
    usando asyncio.get_event_loop().run_in_executor, que mueve la inferencia a
    un hilo separado sin bloquear el loop.

    Atributos:
        _modelo: Instancia cargada del modelo SentenceTransformer
        _nombre_modelo: Identificador del modelo para registro en BD
        _dimension: Dimensión del vector embedding (típicamente 384 para MiniLM)
        _executor: None para usar el ThreadPoolExecutor por defecto de asyncio
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
            self._nombre_modelo: str = ajustes.modelo_texto
            self._modelo: SentenceTransformer = SentenceTransformer(self._nombre_modelo)
            self._dimension: int = ajustes.dimension_vector_texto
        except Exception as e:
            raise RuntimeError(
                f"Error al cargar el modelo de embeddings '{ajustes.modelo_texto}': {e}"
            ) from e

    # ------------------------------------------------------------------
    # Helpers síncronos — se ejecutan en el ThreadPoolExecutor
    # ------------------------------------------------------------------

    def _encode_one(self, texto: str) -> List[float]:
        """
        Codifica un texto de forma síncrona. Solo llamar desde run_in_executor.

        Args:
            texto: Texto ya normalizado (strip + lower).

        Returns:
            Lista de flotantes con el embedding.
        """
        embedding = self._modelo.encode(
            texto,
            convert_to_tensor=False
        )
        return embedding.tolist()

    def _encode_batch(self, textos: List[str]) -> List[List[float]]:
        """
        Codifica una lista de textos de forma síncrona. Solo llamar desde run_in_executor.

        Args:
            textos: Lista de textos ya normalizados.

        Returns:
            Lista de listas de flotantes.
        """
        embeddings = self._modelo.encode(
            textos,
            convert_to_tensor=False,
            batch_size=32,
            show_progress_bar=False
        )
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # Interfaz pública async — delegan al executor para no bloquear el loop
    # ------------------------------------------------------------------

    async def codificar_texto(self, texto: str) -> List[float]:
        """
        Codifica un texto individual en un vector embedding de 384 dimensiones.

        Delega la inferencia síncrona de PyTorch a un hilo separado mediante
        run_in_executor, liberando el event loop de asyncio durante el cómputo.

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

        texto_normalizado = texto.strip().lower()

        if not texto_normalizado:
            raise ValueError("El texto después de normalización está vacío")

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._encode_one,
                texto_normalizado
            )
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error al codificar texto: {e}") from e

    async def codificar_textos(self, textos: List[str]) -> List[List[float]]:
        """
        Codifica una lista de textos en modo batch para máxima eficiencia.

        Delega la inferencia síncrona de PyTorch a un hilo separado mediante
        run_in_executor. Es significativamente más rápido que llamar
        codificar_texto() en un loop, y no bloquea el event loop.

        Args:
            textos: Lista de textos a codificar. Puede estar vacía.
                    Cada elemento se normaliza igual que en codificar_texto().

        Returns:
            Lista de listas de flotantes. Mantiene orden con entrada.

        Raises:
            ValueError: Si algún elemento no es string o está vacío tras normalización
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

        # Validar y normalizar en el hilo principal (operación liviana)
        textos_normalizados: List[str] = []
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

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._encode_batch,
                textos_normalizados
            )
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error al codificar batch de textos: {e}") from e

    async def codificar_imagen(self, ruta_imagen: str) -> List[float]:
        """
        No implementado para este codificador.

        CodificadorTexto está diseñado exclusivamente para textos.
        Para imágenes, usa CodificadorImagen que implementa CLIP.

        Raises:
            NotImplementedError: Siempre.
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
        """
        return self._dimension

    async def nombre_modelo(self) -> str:
        """
        Retorna el nombre del modelo de embeddings de texto.

        Returns:
            Nombre del modelo (ej: 'all-MiniLM-L6-v2')
        """
        return self._nombre_modelo