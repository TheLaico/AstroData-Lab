"""
Herramientas MCP para el flujo RAG minimo de AstroData Lab.

Esta capa actua como adaptador entre el servidor MCP y los repositorios. La
respuesta generada es extractiva/sintetica para mantener el MVP sin dependencia
obligatoria de un LLM externo.
"""

from typing import Any, Dict, List, Optional

from mcp.types import Tool

from config.ajustes import ajustes
from database.repositorio_consultas import RepositorioConsultas
from database.repositorio_documentos import RepositorioDocumentos
from database.repositorio_objetos import RepositorioObjetos
from models.consulta_entrada_model import ConsultaEntrada


def _objeto_a_dict(objeto: Any) -> Dict[str, Any]:
    if objeto is None:
        return {}
    if hasattr(objeto, "model_dump"):
        return objeto.model_dump()
    return {
        k: v
        for k, v in vars(objeto).items()
        if not k.startswith("_")
    } if hasattr(objeto, "__dict__") else {}


class ToolsConsultaRAG:
    """Tools para consultas RAG y recuperacion de contexto de objetos."""

    def __init__(self, codificador: Any) -> None:
        self.codificador = codificador
        self.repo_consultas = RepositorioConsultas()
        self.repo_documentos = RepositorioDocumentos()
        self.repo_objetos = RepositorioObjetos()

    def obtener_definiciones_tools(self) -> List[Tool]:
        return [
            Tool(
                name="rag_query",
                description="Ejecuta una consulta RAG sobre documentos astronomicos.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texto_pregunta": {"type": "string"},
                        "id_usuario": {"type": "integer", "default": 1},
                        "top_k": {"type": "integer", "default": ajustes.top_k},
                        "estrategia_chunking": {"type": "string"},
                    },
                    "required": ["texto_pregunta"],
                },
            ),
            Tool(
                name="obtener_contexto_objeto",
                description="Obtiene contexto relacional y documental de un objeto astronomico.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_objeto": {"type": ["integer", "null"]},
                        "nombre": {"type": ["string", "null"]},
                    },
                },
            ),
        ]

    async def rag_query(
        self,
        texto_pregunta: str,
        id_usuario: int = 1,
        top_k: int = 5,
        estrategia_chunking: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(texto_pregunta, str) or not texto_pregunta.strip():
            return {"error": "La pregunta no puede estar vacia."}
        if not isinstance(top_k, int) or top_k <= 0:
            return {"error": "top_k debe ser un entero positivo."}

        try:
            consulta = await self.repo_consultas.registrar_consulta(
                ConsultaEntrada(texto_pregunta=texto_pregunta.strip(), id_usuario=id_usuario)
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
            respuesta = self._generar_respuesta_extractiva(texto_pregunta, chunks_ordenados)
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
                "respuesta_generada": respuesta,
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
                "objeto": _objeto_a_dict(objeto),
                "documentos": [_objeto_a_dict(doc) for doc in documentos],
                "caracteristicas_ambientales": [
                    _objeto_a_dict(c) for c in caracteristicas
                ],
            }
        except Exception as exc:
            return {"error": f"Error al obtener contexto del objeto: {exc}"}

    def _generar_respuesta_extractiva(
        self,
        pregunta: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
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
