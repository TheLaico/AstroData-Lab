"""Adaptadores MCP de busqueda semantica."""

from typing import Any, Dict, List, Optional

from mcp.types import Tool

from config.ajustes import ajustes
from database.repositorio_documentos import RepositorioDocumentos
from database.repositorio_objetos import RepositorioObjetos
from services.semantic_search_service import BusquedaSemanticaService


class BusquedaSematica:
    """Adaptador MCP para busquedas por similitud."""

    def __init__(self, codificador: Any, codificador_imagen: Any = None) -> None:
        self.service = BusquedaSemanticaService(
            codificador=codificador,
            repo_documentos=RepositorioDocumentos(),
            repo_objetos=RepositorioObjetos(),
            codificador_imagen=codificador_imagen,
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
            ),
            Tool(
                name="buscar_imagenes_por_descripcion",
                description="Busca imagenes astronomicas por una frase del usuario usando CLIP, por ejemplo planetas con anillos o galaxias espirales.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "descripcion": {"type": "string"},
                        "top_k": {"type": ["integer", "string"], "default": ajustes.top_k},
                    },
                    "required": ["descripcion"],
                },
            ),
            Tool(
                name="buscar_imagenes_similares",
                description="Busca imagenes similares en la base de datos a partir de una ruta local o una imagen en base64.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ruta_imagen": {"type": ["string", "null"]},
                        "imagen_base64": {"type": ["string", "null"]},
                        "extension": {"type": "string", "default": "png"},
                        "top_k": {"type": ["integer", "string"], "default": ajustes.top_k},
                    },
                },
            ),
            Tool(
                name="obtener_info_objeto_por_imagen",
                description="Identifica la imagen astronomica mas similar y devuelve textualmente la informacion del objeto asociado.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ruta_imagen": {"type": ["string", "null"]},
                        "imagen_base64": {"type": ["string", "null"]},
                        "extension": {"type": "string", "default": "png"},
                        "top_k": {"type": ["integer", "string"], "default": 3},
                    },
                },
            ),
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

    async def buscar_imagenes_por_descripcion(
        self,
        descripcion: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return await self.buscar_imagenes_semanticas(descripcion=descripcion, top_k=top_k)

    async def buscar_imagenes_similares(
        self,
        ruta_imagen: Optional[str] = None,
        imagen_base64: Optional[str] = None,
        extension: str = "png",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return await self.service.buscar_imagenes_similares_por_imagen(
            ruta_imagen=ruta_imagen,
            imagen_base64=imagen_base64,
            extension=extension,
            top_k=top_k,
        )

    async def obtener_info_objeto_por_imagen(
        self,
        ruta_imagen: Optional[str] = None,
        imagen_base64: Optional[str] = None,
        extension: str = "png",
        top_k: int = 3,
    ) -> Dict[str, Any]:
        return await self.service.obtener_info_objeto_por_imagen(
            ruta_imagen=ruta_imagen,
            imagen_base64=imagen_base64,
            extension=extension,
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
