"""Servicio de aplicacion para busqueda semantica."""

from typing import Any, Dict, Optional

from services.utils import to_int


class BusquedaSemanticaService:
    def __init__(self, codificador: Any, repo_documentos: Any, repo_objetos: Any) -> None:
        self.codificador = codificador
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
            vector = await self.codificador.codificar_texto(descripcion)
            if not hasattr(self.repo_documentos, "buscar_imagenes_similares"):
                return {
                    "descripcion": descripcion.strip(),
                    "imagenes": [],
                    "total": 0,
                    "advertencia": "El repositorio no implementa busqueda de imagenes.",
                }
            filas = await self.repo_documentos.buscar_imagenes_similares(vector, top_k)
            imagenes = sorted(
                filas,
                key=lambda i: float(i.get("similitud", 0.0)),
                reverse=True,
            )[:top_k]
            return {"descripcion": descripcion.strip(), "imagenes": imagenes, "total": len(imagenes)}
        except Exception as exc:
            return {"error": f"Error en busqueda semantica de imagenes: {exc}"}

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
