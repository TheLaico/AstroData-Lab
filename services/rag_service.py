"""Servicio de aplicacion para consultas RAG y contexto de objetos."""

from typing import Any, Dict, List, Optional

from models.consulta_entrada_model import ConsultaEntrada
from services.utils import dump_model, to_optional_int


class ConsultaRAGService:
    """Orquesta registro de consultas, embeddings y recuperacion de chunks."""

    def __init__(
        self,
        codificador: Any,
        repo_consultas: Any,
        repo_documentos: Any,
        repo_objetos: Any,
    ) -> None:
        self.codificador = codificador
        self.repo_consultas = repo_consultas
        self.repo_documentos = repo_documentos
        self.repo_objetos = repo_objetos

    async def rag_query(
        self,
        texto_pregunta: str,
        id_usuario: int = 1,
        top_k: int = 5,
        estrategia_chunking: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(texto_pregunta, str) or not texto_pregunta.strip():
            return {"error": "La pregunta no puede estar vacia."}
        top_k = int(top_k)
        if top_k <= 0:
            return {"error": "top_k debe ser un entero positivo."}

        try:
            consulta = await self.repo_consultas.registrar_consulta(
                ConsultaEntrada(texto_pregunta=texto_pregunta.strip(), id_usuario=int(id_usuario))
            )
            vector = await self.codificador.codificar_texto(texto_pregunta)
            modelo = await self.codificador.nombre_modelo()
            await self.repo_consultas.guardar_embedding_consulta(
                consulta.id_consulta,
                vector,
                modelo,
            )
            chunks = await self.repo_documentos.buscar_chunks_similares(
                vector,
                top_k,
                estrategia_chunking,
            )
            chunks_ordenados = sorted(
                chunks,
                key=lambda c: float(c.get("similitud", c.get("puntuacion_similitud", 0.0))),
                reverse=True,
            )[:top_k]
            fuentes = [
                {
                    "id_doc": c.get("id_doc"),
                    "titulo": c.get("titulo"),
                    "chunk_id": c.get("chunk_id"),
                    "similitud": c.get("similitud", c.get("puntuacion_similitud")),
                }
                for c in chunks_ordenados
            ]
            return {
                "id_consulta": consulta.id_consulta,
                "pregunta": texto_pregunta.strip(),
                "respuesta_generada": self._generar_respuesta_extractiva(chunks_ordenados),
                "chunks_recuperados": chunks_ordenados,
                "fuentes": fuentes,
                "metadata": {
                    "top_k": top_k,
                    "modelo_embedding": modelo,
                    "estrategia_chunking": estrategia_chunking,
                    "modo_respuesta": "extractivo_mvp",
                },
            }
        except Exception as exc:
            return {"error": f"Error al ejecutar consulta RAG: {exc}"}

    async def obtener_contexto_objeto(
        self,
        id_objeto: Optional[int] = None,
        nombre: Optional[str] = None,
    ) -> Dict[str, Any]:
        id_objeto = to_optional_int(id_objeto, "id_objeto")
        if id_objeto is None and not (isinstance(nombre, str) and nombre.strip()):
            return {"error": "Debe proporcionar id_objeto o nombre."}

        try:
            objeto = None
            if id_objeto is not None:
                objeto = await self.repo_objetos.obtener_objeto_por_id(id_objeto)
            else:
                objeto = await self.repo_objetos.obtener_objeto_por_nombre(nombre or "")

            if objeto is None:
                return {"error": "No se encontro el objeto astronomico solicitado."}

            oid = getattr(objeto, "id_objeto", id_objeto)
            documentos = []
            if hasattr(self.repo_documentos, "listar_documentos_por_objeto"):
                documentos = await self.repo_documentos.listar_documentos_por_objeto(oid)
            elif hasattr(self.repo_objetos, "listar_documentos_por_objeto"):
                documentos = await self.repo_objetos.listar_documentos_por_objeto(oid)

            caracteristicas = []
            if hasattr(self.repo_objetos, "obtener_caracteristicas_ambientales"):
                try:
                    caracteristicas = await self.repo_objetos.obtener_caracteristicas_ambientales(oid)
                except Exception:
                    caracteristicas = []

            return {
                "objeto": dump_model(objeto),
                "documentos": [dump_model(doc) for doc in documentos],
                "caracteristicas_ambientales": [dump_model(c) for c in caracteristicas],
            }
        except Exception as exc:
            return {"error": f"Error al obtener contexto del objeto: {exc}"}

    def _generar_respuesta_extractiva(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return (
                "No se encontraron fragmentos relevantes en la base de conocimiento "
                "para responder la pregunta."
            )
        partes = []
        for chunk in chunks[:3]:
            contenido = chunk.get("contenido") or chunk.get("contenido_chunk")
            titulo = chunk.get("titulo", "fuente sin titulo")
            if contenido:
                partes.append(f"{titulo}: {contenido}")
            else:
                partes.append(f"{titulo} (chunk {chunk.get('chunk_id')})")
        return "Respuesta basada en contexto recuperado: " + " | ".join(partes)
