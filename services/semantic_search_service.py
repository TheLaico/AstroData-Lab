"""Servicio de aplicacion para busqueda semantica."""

import base64
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from services.utils import to_int


class BusquedaSemanticaService:
    def __init__(
        self,
        codificador: Any,
        repo_documentos: Any,
        repo_objetos: Any,
        codificador_imagen: Any = None,
    ) -> None:
        self.codificador = codificador
        self.codificador_imagen = codificador_imagen or codificador
        self.repo_documentos = repo_documentos
        self.repo_objetos = repo_objetos

    async def buscar_documentos_semanticos(
        self,
        consulta: str,
        top_k: int = 5,
        estrategia_chunking: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(consulta, str) or not consulta.strip():
            return {"error": "La consulta no puede estar vacia."}
        top_k = int(top_k)
        if top_k <= 0:
            return {"error": "top_k debe ser un entero positivo."}

        try:
            vector = await self.codificador.codificar_texto(consulta)
            filas = await self.repo_documentos.buscar_chunks_similares(
                vector,
                top_k,
                estrategia_chunking,
            )
            documentos = [self._normalizar_documento_resultado(fila) for fila in filas]
            documentos.sort(key=lambda d: d["puntuacion_similitud"], reverse=True)
            return {
                "consulta": consulta.strip(),
                "documentos": documentos[:top_k],
                "total": min(len(documentos), top_k),
            }
        except Exception as exc:
            return {"error": f"Error en busqueda semantica de documentos: {exc}"}

    async def buscar_imagenes_semanticas(self, descripcion: str, top_k: int = 5) -> Dict[str, Any]:
        if not isinstance(descripcion, str) or not descripcion.strip():
            return {"error": "La descripcion no puede estar vacia."}
        top_k = int(top_k)
        if top_k <= 0:
            return {"error": "top_k debe ser un entero positivo."}

        try:
            vector = await self.codificador_imagen.codificar_texto(descripcion)
            filas = await self.repo_documentos.buscar_imagenes_similares(vector, top_k)
            imagenes = sorted(
                [self._normalizar_imagen_resultado(fila) for fila in filas],
                key=lambda i: float(i.get("similitud", 0.0)),
                reverse=True,
            )[:top_k]
            return {"descripcion": descripcion.strip(), "imagenes": imagenes, "total": len(imagenes)}
        except Exception as exc:
            return {"error": f"Error en busqueda semantica de imagenes: {exc}"}

    async def buscar_imagenes_similares_por_imagen(
        self,
        ruta_imagen: Optional[str] = None,
        imagen_base64: Optional[str] = None,
        top_k: int = 5,
        extension: str = "png",
    ) -> Dict[str, Any]:
        top_k = int(top_k)
        if top_k <= 0:
            return {"error": "top_k debe ser un entero positivo."}
        if not ruta_imagen and not imagen_base64:
            return {"error": "Debe proporcionar ruta_imagen o imagen_base64."}

        ruta_temporal = None
        try:
            ruta_a_codificar = ruta_imagen
            if imagen_base64:
                ruta_temporal = self._guardar_imagen_base64_temporal(imagen_base64, extension)
                ruta_a_codificar = str(ruta_temporal)

            vector = await self.codificador_imagen.codificar_imagen(str(ruta_a_codificar))
            filas = await self.repo_documentos.buscar_imagenes_similares(vector, top_k)
            imagenes = [
                self._normalizar_imagen_resultado(fila)
                for fila in sorted(filas, key=lambda i: float(i.get("similitud", 0.0)), reverse=True)[:top_k]
            ]
            return {
                "imagen_consulta": "imagen_base64" if imagen_base64 else str(ruta_imagen),
                "imagenes": imagenes,
                "total": len(imagenes),
            }
        except Exception as exc:
            return {"error": f"Error en busqueda por imagen similar: {exc}"}
        finally:
            if ruta_temporal is not None:
                try:
                    ruta_temporal.unlink(missing_ok=True)
                except Exception:
                    pass

    async def obtener_info_objeto_por_imagen(
        self,
        ruta_imagen: Optional[str] = None,
        imagen_base64: Optional[str] = None,
        top_k: int = 3,
        extension: str = "png",
    ) -> Dict[str, Any]:
        resultado = await self.buscar_imagenes_similares_por_imagen(
            ruta_imagen=ruta_imagen,
            imagen_base64=imagen_base64,
            top_k=top_k,
            extension=extension,
        )
        if "error" in resultado:
            return resultado

        imagenes = resultado.get("imagenes", [])
        coincidencia = next((img for img in imagenes if img.get("objeto")), None)
        if coincidencia is None:
            return {
                "respuesta_textual": "No encontre un objeto astronomico asociado a las imagenes mas similares.",
                "coincidencias": imagenes,
            }

        objeto = coincidencia["objeto"]
        nombre = objeto.get("nombre") or "objeto astronomico sin nombre"
        tipo = objeto.get("tipo") or "objeto astronomico"
        descripcion = objeto.get("descripcion_cientifica") or "No hay descripcion cientifica registrada."
        return {
            "respuesta_textual": f"La imagen coincide principalmente con {nombre}, clasificado como {tipo}. {descripcion}",
            "objeto_detectado": objeto,
            "coincidencia_principal": coincidencia,
            "coincidencias": imagenes,
        }

    async def encontrar_planetas_similares(self, id_planeta: int, top_k: int = 5) -> Dict[str, Any]:
        id_planeta = to_int(id_planeta, "id_planeta")
        top_k = int(top_k)
        if id_planeta <= 0:
            return {"error": "id_planeta debe ser un entero positivo."}
        if top_k <= 0:
            return {"error": "top_k debe ser un entero positivo."}

        try:
            planeta = await self.repo_objetos.obtener_objeto_por_id(id_planeta)
            if planeta is None:
                return {"error": "No se encontro el planeta de referencia."}

            descripcion = getattr(planeta, "descripcion_cientifica", None) or getattr(planeta, "nombre", "")
            vector = await self.codificador.codificar_texto(descripcion)
            filas = await self.repo_documentos.buscar_chunks_similares(vector, top_k + 1, "semantic")
            nombre_ref = getattr(planeta, "nombre", "")
            similares = []
            for fila in filas:
                nombre = fila.get("nombre") or fila.get("titulo") or "Planeta sin nombre"
                if nombre == nombre_ref:
                    continue
                similares.append(
                    {
                        "nombre": nombre,
                        "chunk_id": fila.get("chunk_id"),
                        "similitud": float(fila.get("similitud", 0.0)),
                        "fuente": fila.get("titulo"),
                    }
                )
                if len(similares) >= top_k:
                    break
            return {
                "planeta_referencia": {
                    "id_objeto": getattr(planeta, "id_objeto", id_planeta),
                    "nombre": nombre_ref,
                },
                "planetas_similares": similares,
            }
        except Exception as exc:
            return {"error": f"Error al buscar planetas similares: {exc}"}

    def _normalizar_documento_resultado(self, fila: Dict[str, Any]) -> Dict[str, Any]:
        similitud = float(fila.get("similitud", fila.get("puntuacion_similitud", 0.0)))
        return {
            "id_doc": fila.get("id_doc"),
            "titulo": fila.get("titulo"),
            "chunk_id": fila.get("chunk_id"),
            "estrategia_chunking": fila.get("estrategia_chunking", fila.get("estrategia")),
            "contenido": fila.get("contenido"),
            "puntuacion_similitud": similitud,
        }

    def _normalizar_imagen_resultado(self, fila: Dict[str, Any]) -> Dict[str, Any]:
        objeto = None
        if fila.get("id_objeto") is not None:
            objeto = {
                "id_objeto": fila.get("id_objeto"),
                "nombre": fila.get("nombre_objeto"),
                "tipo": fila.get("tipo_objeto"),
                "descripcion_cientifica": fila.get("descripcion_cientifica"),
            }
        documento = None
        if fila.get("id_doc") is not None:
            documento = {
                "id_doc": fila.get("id_doc"),
                "titulo": fila.get("titulo_documento"),
                "fuente": fila.get("fuente_documento"),
            }
        return {
            "id_imagen": fila.get("id_imagen"),
            "ruta_archivo": fila.get("ruta_archivo"),
            "descripcion": fila.get("descripcion"),
            "etiquetas": fila.get("etiquetas"),
            "similitud": float(fila.get("similitud", 0.0)),
            "documento": documento,
            "objeto": objeto,
        }

    def _guardar_imagen_base64_temporal(self, imagen_base64: str, extension: str) -> Path:
        contenido = imagen_base64.strip()
        if "," in contenido and contenido.lower().startswith("data:"):
            contenido = contenido.split(",", 1)[1]
        extension_limpia = (extension or "png").strip().lower().lstrip(".")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension_limpia}") as archivo:
            archivo.write(base64.b64decode(contenido))
            return Path(archivo.name)
