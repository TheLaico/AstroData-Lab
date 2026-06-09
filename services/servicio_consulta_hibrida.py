"""Servicio de consulta híbrida que combina RAG y búsqueda semántica con filtros exactos."""

import asyncio
from typing import Any, Dict, List, Optional

from services.rag_service import ConsultaRAGService
from services.semantic_search_service import BusquedaSemanticaService
from services.utils import dump_model
from database.repositorio_consultas import RepositorioConsultas
from database.repositorio_documentos import RepositorioDocumentos
from database.repositorio_objetos import RepositorioObjetos


class ServicioConsultaHibrida:
    """Orquesta la consulta híbrida sin reimplementar la lógica del repositorio."""

    def __init__(
        self,
        codificador: Any,
        codificador_imagen: Any = None,
    ) -> None:
        self.rag_service = ConsultaRAGService(
            codificador=codificador,
            repo_consultas=RepositorioConsultas(),
            repo_documentos=RepositorioDocumentos(),
            repo_objetos=RepositorioObjetos(),
        )
        self.semantic_service = BusquedaSemanticaService(
            codificador=codificador,
            repo_documentos=RepositorioDocumentos(),
            repo_objetos=RepositorioObjetos(),
            codificador_imagen=codificador_imagen,
        )

    async def consulta_hibrida(
        self,
        texto_pregunta: str,
        filtros: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        alpha: float = 0.7,
    ) -> Dict[str, Any]:
        if not isinstance(texto_pregunta, str) or not texto_pregunta.strip():
            return {"error": "La pregunta no puede estar vacía."}

        try:
            top_k = int(top_k)
            alpha = float(alpha)
        except (TypeError, ValueError):
            return {"error": "top_k debe ser entero y alpha un número."}

        if top_k <= 0:
            return {"error": "top_k debe ser un entero positivo."}
        if alpha < 0.0 or alpha > 1.0:
            return {"error": "alpha debe estar entre 0.0 y 1.0."}

        filtros = filtros or {}
        if not isinstance(filtros, dict):
            return {"error": "filtros debe ser un diccionario de metadatos."}

        try:
            rag_task = asyncio.create_task(
                self.rag_service.rag_query(
                    texto_pregunta=texto_pregunta,
                    id_usuario=1,
                    top_k=top_k,
                )
            )
            sem_task = asyncio.create_task(
                self.semantic_service.buscar_documentos_semanticos(
                    consulta=texto_pregunta,
                    top_k=top_k,
                )
            )

            rag_result, sem_result = await asyncio.gather(rag_task, sem_task)

            if "error" in rag_result:
                return {"error": f"Error en búsqueda RAG: {rag_result['error']}"}
            if "error" in sem_result:
                return {"error": f"Error en búsqueda exacta: {sem_result['error']}"}

            rag_docs = self._extraer_documentos_de_rag(rag_result)
            exact_docs = self._filtrar_documentos_exactos(
                sem_result.get("documentos", []), filtros
            )

            resultados = self._reciprocal_rank_fusion(
                rag_docs=rag_docs,
                exact_docs=exact_docs,
                top_k=top_k,
                alpha=alpha,
            )

            return {
                "texto_pregunta": texto_pregunta.strip(),
                "filtros": filtros,
                "top_k": top_k,
                "alpha": alpha,
                "resultados": resultados,
            }
        except Exception as exc:
            return {"error": f"Error al ejecutar consulta híbrida: {exc}"}

    def _extraer_documentos_de_rag(self, rag_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        documentos: Dict[Any, Dict[str, Any]] = {}
        for chunk in rag_result.get("chunks_recuperados", []):
            id_doc = chunk.get("id_doc")
            if id_doc is None:
                continue
            similitud = float(chunk.get("similitud", 0.0))
            actual = documentos.get(id_doc)
            if actual is None or similitud > actual["similitud"]:
                documentos[id_doc] = {
                    "id_doc": id_doc,
                    "titulo": chunk.get("titulo"),
                    "chunk_id": chunk.get("chunk_id"),
                    "contenido": chunk.get("contenido"),
                    "similitud": similitud,
                    "estrategia_chunking": chunk.get("estrategia"),
                }
        return list(documentos.values())

    def _filtrar_documentos_exactos(
        self,
        documentos: List[Dict[str, Any]],
        filtros: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not filtros:
            return documentos

        filtros_limpios = {}
        for clave, valor in filtros.items():
            if valor is None:
                continue
            filtros_limpios[str(clave).strip()] = str(valor).strip().lower()

        if not filtros_limpios:
            return documentos

        exactos: List[Dict[str, Any]] = []
        for doc in documentos:
            cumple = True
            for clave, valor in filtros_limpios.items():
                valor_doc = doc.get(clave)
                if valor_doc is None:
                    cumple = False
                    break
                if isinstance(valor_doc, (int, float)):
                    if str(valor_doc) != valor:
                        cumple = False
                        break
                elif str(valor_doc).strip().lower() != valor:
                    cumple = False
                    break
            if cumple:
                exactos.append(doc)

        return exactos

    def _reciprocal_rank_fusion(
        self,
        rag_docs: List[Dict[str, Any]],
        exact_docs: List[Dict[str, Any]],
        top_k: int,
        alpha: float,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        rag_scores: Dict[Any, float] = {
            doc["id_doc"]: 1.0 / (k + idx)
            for idx, doc in enumerate(rag_docs, start=1)
        }
        exact_scores: Dict[Any, float] = {
            doc["id_doc"]: 1.0 / (k + idx)
            for idx, doc in enumerate(exact_docs, start=1)
        }

        all_ids = set(rag_scores) | set(exact_scores)
        fused: List[Dict[str, Any]] = []

        for id_doc in all_ids:
            rag_item = next((doc for doc in rag_docs if doc["id_doc"] == id_doc), {})
            exact_item = next((doc for doc in exact_docs if doc["id_doc"] == id_doc), {})
            rag_score = rag_scores.get(id_doc, 0.0)
            exact_score = exact_scores.get(id_doc, 0.0)
            hybrid_score = alpha * rag_score + (1.0 - alpha) * exact_score

            resultado = {
                "id_doc": id_doc,
                "titulo": rag_item.get("titulo") or exact_item.get("titulo"),
                "contenido": rag_item.get("contenido") or exact_item.get("contenido"),
                "puntuacion_rag": rag_score,
                "puntuacion_exacta": exact_score,
                "hybrid_score": hybrid_score,
                "fuente": exact_item.get("fuente") or rag_item.get("fuente"),
                "estrategia_chunking": rag_item.get("estrategia_chunking") or exact_item.get("estrategia_chunking"),
                "origen": self._detectar_origen(rag_score, exact_score),
            }
            fused.append(resultado)

        fused.sort(
            key=lambda item: (
                item["hybrid_score"],
                item["puntuacion_rag"],
                item["puntuacion_exacta"],
            ),
            reverse=True,
        )
        return fused[:top_k]

    def _detectar_origen(self, rag_score: float, exact_score: float) -> str:
        if rag_score > 0.0 and exact_score > 0.0:
            return "ambos"
        if rag_score > 0.0:
            return "rag"
        if exact_score > 0.0:
            return "exacto"
        return "desconocido"
