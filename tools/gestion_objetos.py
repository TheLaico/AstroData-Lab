"""Tools MCP para gestion de objetos astronomicos y consultas estructuradas."""

from typing import Any, Dict, List, Optional

from mcp.types import Tool

from database.repositorio_documentos import RepositorioDocumentos
from database.repositorio_objetos import RepositorioObjetos
from database.repositorio_observaciones import RepositorioObservaciones
from services.objetos_service import GestionObjetosService


class GestionObjetos:
    """Adaptador MCP para CRUD astronomico basico."""

    TIPOS_VALIDOS = {"galaxia", "sistema_estelar", "estrella", "planeta", "luna"}

    def __init__(self, codificador: Any) -> None:
        self.service = GestionObjetosService(
            codificador=codificador,
            repo_objetos=RepositorioObjetos(),
            repo_documentos=RepositorioDocumentos(),
            repo_observaciones=RepositorioObservaciones(),
        )

    def obtener_definiciones_tools(self) -> List[Tool]:
        return [
            Tool(
                name="crear_objeto_astronomico",
                description="Crea un objeto astronomico basico.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "nombre": {"type": "string"},
                        "tipo": {"type": "string", "enum": sorted(self.TIPOS_VALIDOS)},
                        "descripcion_cientifica": {"type": ["string", "null"]},
                        "atributos": {"type": "object"},
                    },
                    "required": ["nombre", "tipo"],
                },
            ),
            Tool(
                name="obtener_objeto_astronomico",
                description="Obtiene un objeto astronomico por id o nombre.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_objeto": {"type": ["integer", "string", "null"]},
                        "nombre": {"type": ["string", "null"]},
                    },
                },
            ),
            Tool(
                name="actualizar_objeto_astronomico",
                description="Actualiza campos basicos de un objeto astronomico.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_objeto": {"type": ["integer", "string"]},
                        "campos": {"type": "object"},
                    },
                    "required": ["id_objeto", "campos"],
                },
            ),
            Tool(
                name="eliminar_objeto_astronomico",
                description="Elimina un objeto astronomico por id.",
                inputSchema={
                    "type": "object",
                    "properties": {"id_objeto": {"type": ["integer", "string"]}},
                    "required": ["id_objeto"],
                },
            ),
            Tool(
                name="listar_planetas_habitables",
                description="Lista planetas con puntaje de habitabilidad minimo.",
                inputSchema={
                    "type": "object",
                    "properties": {"puntaje_minimo": {"type": ["number", "string"], "default": 0.7}},
                },
            ),
            Tool(
                name="crear_documento_con_embeddings",
                description="Crea un documento y genera un embedding textual.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "titulo": {"type": "string"},
                        "contenido_texto": {"type": "string"},
                        "id_objeto": {"type": ["integer", "string", "null"]},
                        "idioma": {"type": "string", "default": "es"},
                        "fuente": {"type": ["string", "null"]},
                        "estrategia_chunking": {"type": "string", "default": "sentence"},
                    },
                    "required": ["titulo", "contenido_texto"],
                },
            ),
            Tool(
                name="crear_imagen_con_embedding",
                description="Registra una imagen astronomica.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ruta_archivo": {"type": "string"},
                        "descripcion": {"type": ["string", "null"]},
                        "etiquetas": {"type": ["array", "null"], "items": {"type": "string"}},
                        "id_doc": {"type": ["integer", "string", "null"]},
                    },
                    "required": ["ruta_archivo"],
                },
            ),
            Tool(
                name="crear_telescopio",
                description="Crea un telescopio.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "nombre": {"type": "string"},
                        "tipo": {"type": ["string", "null"]},
                        "ubicacion": {"type": ["string", "null"]},
                    },
                    "required": ["nombre"],
                },
            ),
            Tool(
                name="obtener_telescopio",
                description="Obtiene un telescopio por id.",
                inputSchema={
                    "type": "object",
                    "properties": {"id_telescopio": {"type": ["integer", "string"]}},
                    "required": ["id_telescopio"],
                },
            ),
            Tool(
                name="listar_telescopios",
                description="Lista telescopios registrados.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="crear_observacion",
                description="Crea una observacion astronomica.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_telescopio": {"type": ["integer", "string"]},
                        "id_objeto": {"type": ["integer", "string"]},
                        "descripcion": {"type": ["string", "null"]},
                        "fecha": {"type": ["string", "null"]},
                    },
                    "required": ["id_telescopio", "id_objeto"],
                },
            ),
            Tool(
                name="listar_observaciones_por_objeto",
                description="Lista observaciones asociadas a un objeto astronomico.",
                inputSchema={
                    "type": "object",
                    "properties": {"id_objeto": {"type": ["integer", "string"]}},
                    "required": ["id_objeto"],
                },
            ),
            Tool(
                name="listar_observaciones_por_telescopio",
                description="Lista observaciones realizadas por un telescopio.",
                inputSchema={
                    "type": "object",
                    "properties": {"id_telescopio": {"type": ["integer", "string"]}},
                    "required": ["id_telescopio"],
                },
            ),
        ]

    async def crear_objeto_astronomico(
        self,
        nombre: str,
        tipo: str,
        descripcion_cientifica: Optional[str] = None,
        atributos: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await self.service.crear_objeto_astronomico(
            nombre=nombre,
            tipo=tipo,
            descripcion_cientifica=descripcion_cientifica,
            atributos=atributos,
        )

    async def obtener_objeto_astronomico(
        self,
        id_objeto: Optional[int] = None,
        nombre: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.service.obtener_objeto_astronomico(id_objeto=id_objeto, nombre=nombre)

    async def actualizar_objeto_astronomico(
        self,
        id_objeto: int,
        campos: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self.service.actualizar_objeto_astronomico(id_objeto=id_objeto, campos=campos)

    async def eliminar_objeto_astronomico(self, id_objeto: int) -> Dict[str, Any]:
        return await self.service.eliminar_objeto_astronomico(id_objeto=id_objeto)

    async def listar_planetas_habitables(self, puntaje_minimo: float = 0.7) -> Dict[str, Any]:
        return await self.service.listar_planetas_habitables(puntaje_minimo=puntaje_minimo)

    async def crear_documento_con_embeddings(
        self,
        titulo: str,
        contenido_texto: str,
        id_objeto: Optional[int] = None,
        idioma: str = "es",
        fuente: Optional[str] = None,
        estrategia_chunking: str = "sentence",
    ) -> Dict[str, Any]:
        return await self.service.crear_documento_con_embeddings(
            titulo=titulo,
            contenido_texto=contenido_texto,
            id_objeto=id_objeto,
            idioma=idioma,
            fuente=fuente,
            estrategia_chunking=estrategia_chunking,
        )

    async def crear_imagen_con_embedding(
        self,
        ruta_archivo: str,
        descripcion: Optional[str] = None,
        etiquetas: Optional[List[str]] = None,
        id_doc: Optional[int] = None,
    ) -> Dict[str, Any]:
        return await self.service.crear_imagen_con_embedding(
            ruta_archivo=ruta_archivo,
            descripcion=descripcion,
            etiquetas=etiquetas,
            id_doc=id_doc,
        )

    async def crear_telescopio(self, nombre: str, tipo: Optional[str] = None, ubicacion: Optional[str] = None) -> Dict[str, Any]:
        return await self.service.crear_telescopio(nombre=nombre, tipo=tipo, ubicacion=ubicacion)

    async def obtener_telescopio(self, id_telescopio: int) -> Dict[str, Any]:
        return await self.service.obtener_telescopio(id_telescopio=id_telescopio)

    async def listar_telescopios(self) -> Dict[str, Any]:
        return await self.service.listar_telescopios()

    async def crear_observacion(
        self,
        id_telescopio: int,
        id_objeto: int,
        descripcion: Optional[str] = None,
        fecha: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.service.crear_observacion(
            id_telescopio=id_telescopio,
            id_objeto=id_objeto,
            descripcion=descripcion,
            fecha=fecha,
        )

    async def listar_observaciones_por_objeto(self, id_objeto: int) -> Dict[str, Any]:
        return await self.service.listar_observaciones_por_objeto(id_objeto=id_objeto)

    async def listar_observaciones_por_telescopio(self, id_telescopio: int) -> Dict[str, Any]:
        return await self.service.listar_observaciones_por_telescopio(id_telescopio=id_telescopio)
