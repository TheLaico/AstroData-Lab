"""
Módulo de codificador de embeddings de imagen para AstroData Lab.

Implementa la interfaz CodificadorBase usando CLIP (Contrastive Language-Image Pre-training),
que proporciona capacidades multimodales para generar embeddings de imágenes y textos en
un espacio vectorial compartido. Se utiliza para vectorizar imágenes astronómicas y buscar
imágenes similares por descripción textual.
"""

from pathlib import Path
from typing import List
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
from config.ajustes import ajustes
from embeddings.interfaz_codificador import CodificadorBase


class CodificadorImagen(CodificadorBase):
    """
    Implementación concreta de CodificadorBase para embeddings de imagen con CLIP.
    
    Utiliza OpenAI's CLIP (vision-language model) para generar vectores densos
    que capturan contenido visual de imágenes astronómicas. Soporta búsqueda
    cruzada: buscar imágenes con descripción textual en el mismo espacio vectorial.
    
    El modelo y procesador se cargan una sola vez al instanciar para eficiencia.
    Se obtiene desde ajustes.modelo_imagen para permitir cambiar modelo sin modificar código.
    
    Atributos:
        _modelo: Instancia cargada del CLIPModel
        _procesador: Instancia de CLIPProcessor para preparar imágenes y textos
        _nombre_modelo: Identificador del modelo para registro en BD
        _dimension: Dimensión del vector embedding (típicamente 512 para CLIP ViT-base)
        _device: Dispositivo torch ('cuda' si GPU disponible, 'cpu' en otro caso)
    """
    
    def __init__(self) -> None:
        """
        Inicializa el codificador cargando el modelo CLIP y su procesador.
        
        Lee el nombre del modelo desde ajustes.modelo_imagen. Descarga de HuggingFace
        la primera vez (cachea en ~/.cache/huggingface). Usa GPU si está disponible.
        
        Raises:
            RuntimeError: Si falla la descarga o carga del modelo
            ValueError: Si el nombre del modelo no existe en HuggingFace
        """
        try:
            self._nombre_modelo = ajustes.modelo_imagen
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Cargar modelo y procesador desde HuggingFace
            self._procesador = CLIPProcessor.from_pretrained(self._nombre_modelo)
            self._modelo = CLIPModel.from_pretrained(self._nombre_modelo)
            self._modelo.to(self._device)
            self._modelo.eval()  # Modo evaluación (sin gradientes)
            
            self._dimension = ajustes.dimension_vector_imagen
        
        except Exception as e:
            raise RuntimeError(
                f"Error al cargar el modelo CLIP '{self._nombre_modelo}': {e}"
            ) from e
    
    async def codificar_imagen(self, ruta_imagen: str) -> List[float]:
        """
        Codifica una imagen en un vector embedding de 512 dimensiones.
        
        Carga la imagen desde archivo (soporta FITS, PNG, JPEG, TIFF, WebP),
        la procesa con CLIPProcessor y extrae el embedding visual. Valida que
        el archivo exista antes de procesarlo.
        
        Args:
            ruta_imagen: Ruta absoluta o relativa al archivo de imagen.
        
        Returns:
            Lista de 512 flotantes representando características visuales.
        
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato no es soportado
            RuntimeError: Si hay error en la codificación
        
        Example:
            >>> codificador = CodificadorImagen()
            >>> embedding = await codificador.codificar_imagen("/datos/galaxias/m31.jpg")
            >>> len(embedding)
            512
        """
        if not isinstance(ruta_imagen, str):
            raise ValueError("La ruta debe ser una cadena de texto")
        
        try:
            # Validar que el archivo existe
            ruta_path = Path(ruta_imagen)
            if not ruta_path.exists():
                raise FileNotFoundError(
                    f"El archivo de imagen no existe: {ruta_imagen}"
                )
            
            # Verificar que es una imagen válida
            formatos_soportados = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.fits'}
            if ruta_path.suffix.lower() not in formatos_soportados:
                raise ValueError(
                    f"Formato de imagen no soportado: {ruta_path.suffix}. "
                    f"Formatos válidos: {', '.join(formatos_soportados)}"
                )
            
            # Cargar imagen
            imagen = Image.open(ruta_imagen).convert('RGB')
            
            # Procesar con CLIP
            entradas = self._procesador(
                images=imagen,
                return_tensors='pt',
                padding=True
            )
            entradas = {k: v.to(self._device) for k, v in entradas.items()}
            
            # Generar embedding de imagen
            with torch.no_grad():
                salida = self._modelo(**entradas)
                embedding_imagen = salida.image_embeds[0]
            
            # Normalizar y convertir a List[float]
            embedding_normalizado = embedding_imagen / embedding_imagen.norm()
            return embedding_normalizado.cpu().tolist()
        
        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Error al codificar imagen: {e}"
            ) from e
    
    async def codificar_texto(self, texto: str) -> List[float]:
        """
        Codifica un texto en el espacio vectorial CLIP (512 dimensiones).
        
        Permite búsqueda cruzada: buscar imágenes similares a una descripción textual
        comparando este embedding de texto con embeddings de imágenes. El texto se
        normaliza (strip, lowercase) antes de procesar.
        
        Args:
            texto: Descripción textual de una imagen (ej: "galaxia espiral azul")
        
        Returns:
            Lista de 512 flotantes en el mismo espacio que embeddings de imagen.
        
        Raises:
            ValueError: Si el texto está vacío
            RuntimeError: Si hay error en la codificación
        
        Example:
            >>> codificador = CodificadorImagen()
            >>> embedding = await codificador.codificar_texto("galaxia espiral roja")
            >>> len(embedding)
            512
        """
        if not texto or not isinstance(texto, str):
            raise ValueError("El texto debe ser una cadena no vacía")
        
        try:
            # Normalizar el texto
            texto_normalizado = texto.strip().lower()
            
            if not texto_normalizado:
                raise ValueError("El texto después de normalización está vacío")
            
            # Procesar con CLIP
            entradas = self._procesador(
                text=texto_normalizado,
                return_tensors='pt',
                padding=True
            )
            entradas = {k: v.to(self._device) for k, v in entradas.items()}
            
            # Generar embedding de texto
            with torch.no_grad():
                salida = self._modelo(**entradas)
                embedding_texto = salida.text_embeds[0]
            
            # Normalizar y convertir a List[float]
            embedding_normalizado = embedding_texto / embedding_texto.norm()
            return embedding_normalizado.cpu().tolist()
        
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Error al codificar texto con CLIP: {e}"
            ) from e
    
    async def codificar_textos(self, textos: List[str]) -> List[List[float]]:
        """
        Codifica una lista de textos en modo batch en el espacio CLIP.
        
        Procesa múltiples descripciones simultáneamente para máxima eficiencia.
        Útil para generar embeddings de múltiples etiquetas o descripciones de imágenes.
        
        Args:
            textos: Lista de descripciones textuales. Puede estar vacía.
                    Cada elemento se normaliza igual que en codificar_texto().
        
        Returns:
            Lista de listas de 512 flotantes. Mantiene orden con entrada.
        
        Raises:
            ValueError: Si algún elemento no es string o está vacío tras normalización
            RuntimeError: Si hay error en la codificación batch
        
        Example:
            >>> codificador = CodificadorImagen()
            >>> descripciones = ["galaxia espiral", "nebulosa roja", "cúmulo estelar"]
            >>> embeddings = await codificador.codificar_textos(descripciones)
            >>> len(embeddings)
            3
            >>> len(embeddings[0])
            512
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
            
            # Procesar batch con CLIP
            entradas = self._procesador(
                text=textos_normalizados,
                return_tensors='pt',
                padding=True
            )
            entradas = {k: v.to(self._device) for k, v in entradas.items()}
            
            # Generar embeddings de texto
            with torch.no_grad():
                salida = self._modelo(**entradas)
                embeddings_texto = salida.text_embeds
            
            # Normalizar y convertir a List[List[float]]
            embeddings_normalizados = embeddings_texto / embeddings_texto.norm(dim=1, keepdim=True)
            return embeddings_normalizados.cpu().tolist()
        
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Error al codificar batch de textos con CLIP: {e}"
            ) from e
    
    async def dimension(self) -> int:
        """
        Retorna la dimensión del vector embedding de imagen CLIP.
        
        Returns:
            512 para openai/clip-vit-base-patch32, 768 para clip-vit-large-patch14, etc.
            El valor específico se obtiene desde ajustes.dimension_vector_imagen.
        
        Example:
            >>> codificador = CodificadorImagen()
            >>> await codificador.dimension()
            512
        """
        return self._dimension
    
    async def nombre_modelo(self) -> str:
        """
        Retorna el nombre del modelo CLIP utilizado.
        
        Se registra en tabla Embedding_Imagen para:
        - Rastrear qué modelo generó cada embedding visual
        - Permitir múltiples versiones de embeddings por imagen
        - Comparar calidad entre modelos CLIP diferentes
        
        Returns:
            Nombre del modelo (ej: 'openai/clip-vit-base-patch32')
        
        Example:
            >>> codificador = CodificadorImagen()
            >>> await codificador.nombre_modelo()
            'openai/clip-vit-base-patch32'
        """
        return self._nombre_modelo
