"""
Módulo de codificador de embeddings de imagen para AstroData Lab.

Implementa la interfaz CodificadorBase usando CLIP (Contrastive Language-Image
Pre-training). Se utiliza para vectorizar imágenes astronómicas y buscar
imágenes similares por descripción textual.

CAMBIO v2: todos los métodos de inferencia usan run_in_executor para no
bloquear el event loop de asyncio durante el cómputo síncrono de PyTorch.
"""

import asyncio
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config.ajustes import ajustes
from embeddings.interfaz_codificador import CodificadorBase


class CodificadorImagen(CodificadorBase):
    """
    Implementación concreta de CodificadorBase para embeddings de imagen con CLIP.

    Utiliza OpenAI's CLIP para generar vectores densos que capturan contenido
    visual de imágenes astronómicas.

    IMPORTANTE: CLIP corre sobre PyTorch síncrono. Todos los métodos de
    codificación delegan al ThreadPoolExecutor por defecto de asyncio mediante
    run_in_executor para no bloquear el event loop del servidor MCP.

    Atributos:
        _modelo: Instancia cargada del CLIPModel
        _procesador: Instancia de CLIPProcessor
        _nombre_modelo: Identificador del modelo para registro en BD
        _dimension: Dimensión del vector embedding (512 para CLIP ViT-base)
        _device: Dispositivo torch ('cuda' o 'cpu')
    """

    def __init__(self) -> None:
        """
        Inicializa el codificador cargando CLIP y su procesador.

        Raises:
            RuntimeError: Si falla la carga del modelo
        """
        try:
            self._nombre_modelo: str = ajustes.modelo_imagen
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._procesador: CLIPProcessor = CLIPProcessor.from_pretrained(self._nombre_modelo)
            self._modelo: CLIPModel = CLIPModel.from_pretrained(self._nombre_modelo)
            self._modelo.to(self._device)
            self._modelo.eval()
            self._dimension: int = ajustes.dimension_vector_imagen
        except Exception as e:
            raise RuntimeError(
                f"Error al cargar el modelo CLIP '{ajustes.modelo_imagen}': {e}"
            ) from e

    # ------------------------------------------------------------------
    # Helpers síncronos — se ejecutan en el ThreadPoolExecutor
    # ------------------------------------------------------------------

    def _encode_imagen_sync(self, ruta_imagen: str) -> List[float]:
        """
        Codifica una imagen de forma síncrona. Solo llamar desde run_in_executor.

        Args:
            ruta_imagen: Ruta al archivo de imagen ya validada.

        Returns:
            Lista de flotantes con el embedding normalizado.
        """
        imagen = Image.open(ruta_imagen).convert('RGB')
        entradas = self._procesador(images=imagen, return_tensors='pt', padding=True)
        entradas = {k: v.to(self._device) for k, v in entradas.items()}

        with torch.no_grad():
            salida = self._modelo(**entradas)
            embedding = salida.image_embeds[0]

        embedding_normalizado = embedding / embedding.norm()
        return embedding_normalizado.cpu().tolist()

    def _encode_texto_sync(self, texto: str) -> List[float]:
        """
        Codifica un texto en espacio CLIP de forma síncrona. Solo llamar desde run_in_executor.

        Args:
            texto: Texto ya normalizado.

        Returns:
            Lista de flotantes con el embedding normalizado.
        """
        entradas = self._procesador(text=texto, return_tensors='pt', padding=True)
        entradas = {k: v.to(self._device) for k, v in entradas.items()}

        with torch.no_grad():
            salida = self._modelo(**entradas)
            embedding = salida.text_embeds[0]

        embedding_normalizado = embedding / embedding.norm()
        return embedding_normalizado.cpu().tolist()

    def _encode_textos_sync(self, textos: List[str]) -> List[List[float]]:
        """
        Codifica batch de textos en espacio CLIP de forma síncrona. Solo llamar desde run_in_executor.

        Args:
            textos: Lista de textos ya normalizados.

        Returns:
            Lista de listas de flotantes.
        """
        entradas = self._procesador(text=textos, return_tensors='pt', padding=True)
        entradas = {k: v.to(self._device) for k, v in entradas.items()}

        with torch.no_grad():
            salida = self._modelo(**entradas)
            embeddings = salida.text_embeds

        embeddings_normalizados = embeddings / embeddings.norm(dim=1, keepdim=True)
        return embeddings_normalizados.cpu().tolist()

    # ------------------------------------------------------------------
    # Interfaz pública async — delegan al executor para no bloquear el loop
    # ------------------------------------------------------------------

    async def codificar_imagen(self, ruta_imagen: str) -> List[float]:
        """
        Codifica una imagen en un vector embedding de 512 dimensiones.

        Delega la inferencia de CLIP a un hilo separado para no bloquear
        el event loop de asyncio.

        Args:
            ruta_imagen: Ruta absoluta o relativa al archivo de imagen.

        Returns:
            Lista de 512 flotantes representando características visuales.

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato no es soportado o la ruta es inválida
            RuntimeError: Si hay error en la codificación
        """
        if not isinstance(ruta_imagen, str):
            raise ValueError("La ruta debe ser una cadena de texto")

        ruta_path = Path(ruta_imagen)
        if not ruta_path.exists():
            raise FileNotFoundError(f"El archivo de imagen no existe: {ruta_imagen}")

        formatos_soportados = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.fits'}
        if ruta_path.suffix.lower() not in formatos_soportados:
            raise ValueError(
                f"Formato no soportado: {ruta_path.suffix}. "
                f"Válidos: {', '.join(formatos_soportados)}"
            )

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._encode_imagen_sync,
                ruta_imagen
            )
        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            raise RuntimeError(f"Error al codificar imagen: {e}") from e

    async def codificar_texto(self, texto: str) -> List[float]:
        """
        Codifica un texto en el espacio vectorial CLIP (512 dimensiones).

        Permite búsqueda cruzada texto→imagen. Delega inferencia al executor.

        Args:
            texto: Descripción textual de una imagen.

        Returns:
            Lista de 512 flotantes en el mismo espacio que embeddings de imagen.

        Raises:
            ValueError: Si el texto está vacío
            RuntimeError: Si hay error en la codificación
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
                self._encode_texto_sync,
                texto_normalizado
            )
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error al codificar texto con CLIP: {e}") from e

    async def codificar_textos(self, textos: List[str]) -> List[List[float]]:
        """
        Codifica una lista de textos en modo batch en el espacio CLIP.

        Delega inferencia al executor para no bloquear el event loop.

        Args:
            textos: Lista de descripciones textuales. Puede estar vacía.

        Returns:
            Lista de listas de 512 flotantes.

        Raises:
            ValueError: Si algún elemento no es string o está vacío
            RuntimeError: Si hay error en la codificación batch
        """
        if not textos:
            return []

        textos_normalizados: List[str] = []
        for texto in textos:
            if not isinstance(texto, str):
                raise ValueError(
                    f"Todos los elementos deben ser strings, recibido: {type(texto)}"
                )
            texto_norm = texto.strip().lower()
            if not texto_norm:
                raise ValueError("Ningún texto puede estar vacío después de normalización")
            textos_normalizados.append(texto_norm)

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._encode_textos_sync,
                textos_normalizados
            )
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error al codificar batch de textos con CLIP: {e}") from e

    async def dimension(self) -> int:
        """
        Retorna la dimensión del vector embedding de imagen CLIP.

        Returns:
            512 para openai/clip-vit-base-patch32.
        """
        return self._dimension

    async def nombre_modelo(self) -> str:
        """
        Retorna el nombre del modelo CLIP utilizado.

        Returns:
            Nombre del modelo (ej: 'openai/clip-vit-base-patch32')
        """
        return self._nombre_modelo