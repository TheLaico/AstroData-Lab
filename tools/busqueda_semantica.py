"""Adaptadores MCP de busqueda semantica."""

from typing import Any, Dict, List, Optional

from mcp.types import Tool

from config.ajustes import ajustes
from database.repositorio_documentos import RepositorioDocumentos
from database.repositorio_objetos import RepositorioObjetos
from services.semantic_search_service import BusquedaSemanticaService


class BusquedaSematica:
    """Adaptador MCP para busquedas por similitud."""

    def __init__(self, codificador: Any) -> None:
        self.service = BusquedaSemanticaService(
            codificador=codificador,
            repo_documentos=RepositorioDocumentos(),
            repo_objetos=RepositorioObjetos(),
        )

    def obtener_definiciones_tools(self) -> List[Tool]:
        return [
            Tool(
                name="encontrar_planetas_similares",
                description="Encuentra planetas similares a uno de referencia.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_planeta": {"type": ["integer", "string"]},
                        "top_k": {"type": ["integer", "string"], "default": ajustes.top_k},
                    },
                    "required": ["id_planeta"],
                },
            )
        ]

    async def buscar_documentos_semanticos(
        self,
        consulta: str,
        top_k: int = 5,
        estrategia_chunking: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.service.buscar_documentos_semanticos(
            consulta=consulta,
            top_k=top_k,
            estrategia_chunking=estrategia_chunking,
        )

    async def buscar_imagenes_semanticas(
        self,
        descripcion: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return await self.service.buscar_imagenes_semanticas(
            descripcion=descripcion,
            top_k=top_k,
        )

    async def encontrar_planetas_similares(
        self,
        id_planeta: int,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return await self.service.encontrar_planetas_similares(
            id_planeta=id_planeta,
            top_k=top_k,
        )
