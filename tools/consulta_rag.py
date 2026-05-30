"""Adaptadores MCP para el flujo RAG minimo de AstroData Lab."""

from typing import Any, Dict, List, Optional

from mcp.types import Tool

from config.ajustes import ajustes
from database.repositorio_consultas import RepositorioConsultas
from database.repositorio_documentos import RepositorioDocumentos
from database.repositorio_objetos import RepositorioObjetos
from services.rag_service import ConsultaRAGService


class ToolsConsultaRAG:
    """Tools para consultas RAG y recuperacion de contexto de objetos."""

    def __init__(self, codificador: Any) -> None:
        self.service = ConsultaRAGService(
            codificador=codificador,
            repo_consultas=RepositorioConsultas(),
            repo_documentos=RepositorioDocumentos(),
            repo_objetos=RepositorioObjetos(),
        )

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
        return await self.service.rag_query(
            texto_pregunta=texto_pregunta,
            id_usuario=id_usuario,
            top_k=top_k,
            estrategia_chunking=estrategia_chunking,
        )

    async def obtener_contexto_objeto(
        self,
        id_objeto: Optional[int] = None,
        nombre: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.service.obtener_contexto_objeto(id_objeto=id_objeto, nombre=nombre)
