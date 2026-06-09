"""
Módulo de codificador de embeddings de imagen para AstroData Lab.

Implementa la interfaz CodificadorBase usando CLIP (Contrastive Language-Image
Pre-training). Se utiliza para vectorizar imágenes astronómicas y buscar
imágenes similares por descripción textual.

CAMBIO v2: todos los métodos de inferencia usan run_in_executor para no
bloquear el event loop de asyncio durante el cómputo síncrono de PyTorch.
"""

import asyncio
import tempfile
import urllib.request
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config.ajustes import ajustes
from database.embeddings.interfaz_codificador import CodificadorBase


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
            salida = self._modelo.get_image_features(**entradas)

        embedding = self._extraer_embedding(salida, "imagen")
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
            salida = self._modelo.get_text_features(**entradas)

        embedding = self._extraer_embedding(salida, "texto")
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
            salida = self._modelo.get_text_features(**entradas)

        embeddings = self._extraer_embeddings_batch(salida, "texto")
        embeddings_normalizados = embeddings / embeddings.norm(dim=1, keepdim=True)
        return embeddings_normalizados.cpu().tolist()

    def _extraer_embedding(self, salida, modalidad: str):
        """Normaliza salidas de distintas versiones de Transformers a tensor 1D."""
        embeddings = self._extraer_embeddings_batch(salida, modalidad)
        return embeddings.reshape(-1, embeddings.shape[-1])[0]

    def _extraer_embeddings_batch(self, salida, modalidad: str):
        """Convierte Tensor o ModelOutput de CLIP en tensor batch x dimension."""
        if hasattr(salida, "reshape") and hasattr(salida, "shape"):
            return salida.reshape(-1, salida.shape[-1])

        if hasattr(salida, "pooler_output") and salida.pooler_output is not None:
            embeddings = salida.pooler_output
            embeddings = self._aplicar_proyeccion_si_corresponde(embeddings, modalidad)
            return embeddings.reshape(-1, embeddings.shape[-1])

        if hasattr(salida, "last_hidden_state") and salida.last_hidden_state is not None:
            embeddings = salida.last_hidden_state[:, 0, :]
            embeddings = self._aplicar_proyeccion_si_corresponde(embeddings, modalidad)
            return embeddings.reshape(-1, embeddings.shape[-1])

        raise RuntimeError(f"No se pudo extraer embedding CLIP de {type(salida).__name__}")

    def _obtener_proyeccion(self, modalidad: str):
        if modalidad == "texto":
            return getattr(self._modelo, "text_projection", None)
        proyeccion = getattr(self._modelo, "visual_projection", None)
        if proyeccion is not None:
            return proyeccion
        return getattr(self._modelo, "image_projection", None)

    def _aplicar_proyeccion_si_corresponde(self, embeddings, modalidad: str):
        if embeddings.shape[-1] == self._dimension:
            return embeddings

        proyeccion = self._obtener_proyeccion(modalidad)
        if proyeccion is None:
            return embeddings

        if callable(proyeccion) and hasattr(proyeccion, "weight"):
            entrada_esperada = proyeccion.weight.shape[1]
            if embeddings.shape[-1] == entrada_esperada:
                return proyeccion(embeddings)
            return embeddings

        if hasattr(proyeccion, "shape") and len(proyeccion.shape) == 2:
            entrada_esperada = proyeccion.shape[0]
            if embeddings.shape[-1] == entrada_esperada:
                return embeddings @ proyeccion

        return embeddings

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

        ruta_temporal = None
        if self._es_url(ruta_imagen):
            ruta_temporal = await self._descargar_imagen_temporal(ruta_imagen)
            ruta_a_codificar = str(ruta_temporal)
        else:
            ruta_path = Path(ruta_imagen)
            if not ruta_path.exists():
                raise FileNotFoundError(f"El archivo de imagen no existe: {ruta_imagen}")
            ruta_a_codificar = ruta_imagen

        formatos_soportados = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.fits'}
        ruta_path_validacion = Path(ruta_a_codificar)
        if ruta_path_validacion.suffix.lower() not in formatos_soportados:
            raise ValueError(
                f"Formato no soportado: {ruta_path_validacion.suffix}. "
                f"Válidos: {', '.join(formatos_soportados)}"
            )

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._encode_imagen_sync,
                ruta_a_codificar
            )
        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            raise RuntimeError(f"Error al codificar imagen: {e}") from e
        finally:
            if ruta_temporal is not None:
                try:
                    ruta_temporal.unlink(missing_ok=True)
                except Exception:
                    pass

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

    def _es_url(self, valor: str) -> bool:
        partes = urlparse(valor)
        return partes.scheme in {"http", "https"} and bool(partes.netloc)

    async def _descargar_imagen_temporal(self, url: str) -> Path:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._descargar_imagen_temporal_sync, url)

    def _descargar_imagen_temporal_sync(self, url: str) -> Path:
        suffix = Path(urlparse(url).path).suffix.lower() or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as archivo:
            request = urllib.request.Request(url, headers={"User-Agent": "AstroData-Lab/1.0"})
            with urllib.request.urlopen(request, timeout=30) as respuesta:
                archivo.write(respuesta.read())
            return Path(archivo.name)
