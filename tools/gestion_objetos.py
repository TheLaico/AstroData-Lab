"""Tools MCP para gestion de objetos astronomicos y consultas estructuradas."""

from datetime import date
from typing import Any, Dict, List, Optional

from mcp.types import Tool

from database.repositorio_documentos import RepositorioDocumentos
from database.repositorio_objetos import RepositorioObjetos
from database.repositorio_observaciones import RepositorioObservaciones
from models.documento_model import Documento
from models.imagen_model import Imagen
from models.observacion_model import Observacion
from models.telescopio_model import Telescopio


def _dump(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        try:
            volcado = obj.model_dump()
            if isinstance(volcado, dict):
                return volcado
        except Exception:
            pass
    datos = {}
    for nombre in (
        "id_objeto", "nombre", "descripcion_cientifica", "id_doc", "titulo",
        "id_imagen", "ruta_archivo", "id_telescopio", "tipo", "ubicacion",
        "id_observacion", "id_planeta", "fecha", "descripcion",
    ):
        if hasattr(obj, nombre):
            valor = getattr(obj, nombre)
            if not callable(valor):
                datos[nombre] = valor
    if datos:
        return datos
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"valor": obj}


class GestionObjetos:
    """Adaptador MCP para CRUD astronomico basico."""

    TIPOS_VALIDOS = {"galaxia", "sistema_estelar", "estrella", "planeta", "luna"}

    def __init__(self, codificador: Any) -> None:
        self.codificador = codificador
        self.repo_objetos = RepositorioObjetos()
        self.repo_documentos = RepositorioDocumentos()
        self.repo_observaciones = RepositorioObservaciones()

    def obtener_definiciones_tools(self) -> List[Tool]:
        names = [
            "crear_objeto_astronomico",
            "obtener_objeto_astronomico",
            "actualizar_objeto_astronomico",
            "eliminar_objeto_astronomico",
            "listar_planetas_habitables",
            "crear_documento_con_embeddings",
            "crear_imagen_con_embedding",
            "crear_telescopio",
            "obtener_telescopio",
            "listar_telescopios",
            "crear_observacion",
            "listar_observaciones_por_objeto",
            "listar_observaciones_por_telescopio",
        ]
        return [
            Tool(
                name=name,
                description=f"Herramienta AstroData Lab: {name}.",
                inputSchema={"type": "object", "properties": {}},
            )
            for name in names
        ]

    async def crear_objeto_astronomico(
        self,
        nombre: str,
        tipo: str,
        descripcion_cientifica: Optional[str] = None,
        atributos: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tipo_norm = (tipo or "").strip().lower()
        if tipo_norm not in self.TIPOS_VALIDOS:
            return {"error": f"Tipo de objeto no valido: {tipo}."}
        if not isinstance(nombre, str) or not nombre.strip():
            return {"error": "El nombre del objeto no puede estar vacio."}

        try:
            objeto = await self.repo_objetos.crear_objeto(nombre, descripcion_cientifica)
            embedding_id = None
            if descripcion_cientifica and descripcion_cientifica.strip():
                vector = await self.codificador.codificar_texto(descripcion_cientifica)
                modelo = await self.codificador.nombre_modelo()
                documento_embedding = await self.repo_documentos.crear_documento(
                    Documento(
                        id_doc=-1,
                        titulo=f"Descripcion cientifica de {nombre.strip()}",
                        idioma="es",
                        fecha=date.today(),
                        fuente="Objeto_Astronomico.descripcion_cientifica",
                        contenido_texto=descripcion_cientifica,
                        id_objeto=getattr(objeto, "id_objeto"),
                    )
                )
                embedding_id = await self.repo_documentos.guardar_embedding_texto(
                    getattr(documento_embedding, "id_doc"),
                    0,
                    vector,
                    modelo,
                    "sentence",
                    descripcion_cientifica,
                )
            return {
                **_dump(objeto),
                "tipo": tipo_norm,
                "atributos": atributos or {},
                "embedding_id": embedding_id,
            }
        except Exception as exc:
            return {"error": f"Error al crear objeto astronomico: {exc}"}

    async def obtener_objeto_astronomico(
        self,
        id_objeto: Optional[int] = None,
        nombre: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            if id_objeto is not None:
                objeto = await self.repo_objetos.obtener_objeto_por_id(id_objeto)
            elif nombre:
                objeto = await self.repo_objetos.obtener_objeto_por_nombre(nombre)
            else:
                return {"error": "Debe proporcionar id_objeto o nombre."}
            if objeto is None:
                return {"error": "Objeto astronomico no encontrado."}
            return {"objeto": _dump(objeto)}
        except Exception as exc:
            return {"error": f"Error al obtener objeto astronomico: {exc}"}

    async def actualizar_objeto_astronomico(
        self,
        id_objeto: int,
        campos: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(id_objeto, int) or id_objeto <= 0:
            return {"error": "id_objeto debe ser un entero positivo."}
        if not campos:
            return {"error": "Debe proporcionar campos para actualizar."}

        try:
            embedding_regenerado = False
            if "descripcion_cientifica" in campos:
                nueva = campos["descripcion_cientifica"]
                await self.repo_objetos.actualizar_descripcion(id_objeto, nueva)
                vector = await self.codificador.codificar_texto(nueva)
                modelo = await self.codificador.nombre_modelo()
                documento_embedding = await self.repo_documentos.crear_documento(
                    Documento(
                        id_doc=-1,
                        titulo=f"Descripcion cientifica actualizada {id_objeto}",
                        idioma="es",
                        fecha=date.today(),
                        fuente="Objeto_Astronomico.descripcion_cientifica",
                        contenido_texto=nueva,
                        id_objeto=id_objeto,
                    )
                )
                await self.repo_documentos.guardar_embedding_texto(
                    getattr(documento_embedding, "id_doc"),
                    0,
                    vector,
                    modelo,
                    "sentence",
                    nueva,
                )
                embedding_regenerado = True
            return {
                "id_objeto": id_objeto,
                "campos_actualizados": list(campos.keys()),
                "embedding_regenerado": embedding_regenerado,
            }
        except Exception as exc:
            return {"error": f"Error al actualizar objeto astronomico: {exc}"}

    async def eliminar_objeto_astronomico(self, id_objeto: int) -> Dict[str, Any]:
        if not isinstance(id_objeto, int) or id_objeto <= 0:
            return {"error": "id_objeto debe ser un entero positivo."}
        try:
            existente = await self.repo_objetos.obtener_objeto_por_id(id_objeto)
            if existente is None:
                return {"error": "No existe un objeto astronomico con ese id."}
            eliminado = await self.repo_objetos.eliminar_objeto(id_objeto)
            return {
                "id_objeto": id_objeto,
                "eliminado": bool(eliminado),
                "confirmacion": "Objeto eliminado correctamente." if eliminado else "No se elimino el objeto.",
            }
        except Exception as exc:
            return {"error": f"Error al eliminar objeto astronomico: {exc}"}

    async def listar_planetas_habitables(self, puntaje_minimo: float = 0.7) -> Dict[str, Any]:
        try:
            planetas = await self.repo_objetos.listar_planetas_por_habitabilidad(puntaje_minimo)
            return {"puntaje_minimo": puntaje_minimo, "planetas": [_dump(p) for p in planetas]}
        except Exception as exc:
            return {"error": f"Error al listar planetas habitables: {exc}"}

    async def crear_documento_con_embeddings(
        self,
        titulo: str,
        contenido_texto: str,
        id_objeto: Optional[int] = None,
        idioma: str = "es",
        fuente: Optional[str] = None,
        estrategia_chunking: str = "sentence",
    ) -> Dict[str, Any]:
        try:
            documento = await self.repo_documentos.crear_documento(
                Documento(
                    id_doc=-1,
                    titulo=titulo,
                    idioma=idioma,
                    fecha=date.today(),
                    fuente=fuente,
                    contenido_texto=contenido_texto,
                    id_objeto=id_objeto,
                )
            )
            vector = await self.codificador.codificar_texto(contenido_texto)
            modelo = await self.codificador.nombre_modelo()
            id_embedding = await self.repo_documentos.guardar_embedding_texto(
                documento.id_doc,
                0,
                vector,
                modelo,
                estrategia_chunking,
                contenido_texto,
            )
            return {"documento": _dump(documento), "embeddings": [id_embedding]}
        except Exception as exc:
            return {"error": f"Error al crear documento con embeddings: {exc}"}

    async def crear_imagen_con_embedding(
        self,
        ruta_archivo: str,
        descripcion: Optional[str] = None,
        etiquetas: Optional[List[str]] = None,
        id_doc: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            imagen = await self.repo_documentos.crear_imagen(
                Imagen(
                    id_imagen=-1,
                    ruta_archivo=ruta_archivo,
                    descripcion=descripcion,
                    etiquetas=etiquetas,
                    id_doc=id_doc,
                )
            )
            return {"imagen": _dump(imagen), "embedding_generado": False}
        except Exception as exc:
            return {"error": f"Error al crear imagen: {exc}"}

    async def crear_telescopio(self, nombre: str, tipo: Optional[str] = None, ubicacion: Optional[str] = None) -> Dict[str, Any]:
        try:
            telescopio = await self.repo_observaciones.crear_telescopio(
                Telescopio(id_telescopio=-1, nombre=nombre, tipo=tipo, ubicacion=ubicacion)
            )
            return {"telescopio": _dump(telescopio)}
        except Exception as exc:
            return {"error": f"Error al crear telescopio: {exc}"}

    async def obtener_telescopio(self, id_telescopio: int) -> Dict[str, Any]:
        try:
            telescopio = await self.repo_observaciones.obtener_telescopio(id_telescopio)
            return {"telescopio": _dump(telescopio)} if telescopio else {"error": "Telescopio no encontrado."}
        except Exception as exc:
            return {"error": f"Error al obtener telescopio: {exc}"}

    async def listar_telescopios(self) -> Dict[str, Any]:
        try:
            telescopios = await self.repo_observaciones.listar_telescopios()
            return {"telescopios": [_dump(t) for t in telescopios]}
        except Exception as exc:
            return {"error": f"Error al listar telescopios: {exc}"}

    async def crear_observacion(
        self,
        id_telescopio: int,
        id_objeto: int,
        descripcion: Optional[str] = None,
        fecha: Optional[str | date] = None,
    ) -> Dict[str, Any]:
        try:
            fecha_obs = date.fromisoformat(fecha) if isinstance(fecha, str) else (fecha or date.today())
            observacion = await self.repo_observaciones.crear_observacion(
                Observacion(
                    id_observacion=-1,
                    id_telescopio=id_telescopio,
                    id_objeto=id_objeto,
                    fecha=fecha_obs,
                    descripcion=descripcion,
                )
            )
            return {"observacion": _dump(observacion)}
        except Exception as exc:
            return {"error": f"Error al crear observacion: {exc}"}

    async def listar_observaciones_por_objeto(self, id_objeto: int) -> Dict[str, Any]:
        try:
            obs = await self.repo_observaciones.listar_observaciones_por_objeto(id_objeto)
            return {"observaciones": [_dump(o) for o in obs]}
        except Exception as exc:
            return {"error": f"Error al listar observaciones por objeto: {exc}"}

    async def listar_observaciones_por_telescopio(self, id_telescopio: int) -> Dict[str, Any]:
        try:
            obs = await self.repo_observaciones.listar_observaciones_por_telescopio(id_telescopio)
            return {"observaciones": [_dump(o) for o in obs]}
        except Exception as exc:
            return {"error": f"Error al listar observaciones por telescopio: {exc}"}
