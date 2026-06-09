"""Servicio de aplicacion para gestion de objetos y observaciones."""

import base64
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.documento_model import Documento
from models.imagen_model import Imagen
from services.chunking_service import ChunkingService
from services.utils import dump_model, to_int, to_optional_int


class GestionObjetosService:
    TIPOS_VALIDOS = {"galaxia", "sistema_estelar", "estrella", "planeta", "luna"}

    def __init__(
        self,
        codificador: Any,
        repo_objetos: Any,
        repo_documentos: Any,
        repo_observaciones: Any,
        codificador_imagen: Any = None,
    ) -> None:
        self.codificador = codificador
        self.codificador_imagen = codificador_imagen or codificador
        self.repo_objetos = repo_objetos
        self.repo_documentos = repo_documentos
        self.repo_observaciones = repo_observaciones
        self.chunking = ChunkingService()

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

        objeto = None
        try:
            objeto = await self.repo_objetos.crear_objeto(nombre, descripcion_cientifica)
            embedding_id = None
            if descripcion_cientifica and descripcion_cientifica.strip():
                embedding_id = await self._crear_documento_embedding_descripcion(
                    id_objeto=getattr(objeto, "id_objeto"),
                    titulo=f"Descripcion cientifica de {nombre.strip()}",
                    contenido=descripcion_cientifica,
                )
            return {
                **dump_model(objeto),
                "tipo": tipo_norm,
                "atributos": atributos or {},
                "embedding_id": embedding_id,
            }
        except Exception as exc:
            # Transacción compensatoria: si el embedding falla después de crear
            # el objeto, eliminar el objeto para no dejar registros huérfanos.
            if objeto is not None:
                try:
                    await self.repo_objetos.eliminar_objeto(getattr(objeto, "id_objeto"))
                except Exception:
                    pass  # Si el rollback falla, el error original es prioritario
            return {"error": f"Error al crear objeto astronomico: {exc}"}

    async def obtener_objeto_astronomico(
        self,
        id_objeto: Optional[int] = None,
        nombre: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            if id_objeto is not None:
                objeto = await self.repo_objetos.obtener_objeto_por_id(to_int(id_objeto, "id_objeto"))
            elif nombre:
                objeto = await self.repo_objetos.obtener_objeto_por_nombre(nombre)
            else:
                return {"error": "Debe proporcionar id_objeto o nombre."}
            if objeto is None:
                return {"error": "Objeto astronomico no encontrado."}
            return {"objeto": dump_model(objeto)}
        except Exception as exc:
            return {"error": f"Error al obtener objeto astronomico: {exc}"}

    async def actualizar_objeto_astronomico(self, id_objeto: int, campos: Dict[str, Any]) -> Dict[str, Any]:
        id_objeto = to_int(id_objeto, "id_objeto")
        if id_objeto <= 0:
            return {"error": "id_objeto debe ser un entero positivo."}
        if not campos:
            return {"error": "Debe proporcionar campos para actualizar."}

        try:
            embedding_regenerado = False
            if "descripcion_cientifica" in campos:
                nueva = campos["descripcion_cientifica"]
                await self.repo_objetos.actualizar_descripcion(id_objeto, nueva)
                await self._crear_documento_embedding_descripcion(
                    id_objeto=id_objeto,
                    titulo=f"Descripcion cientifica actualizada {id_objeto}",
                    contenido=nueva,
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
        id_objeto = to_int(id_objeto, "id_objeto")
        if id_objeto <= 0:
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
                "politica_documentos": "Los documentos asociados se conservan y quedan desvinculados del objeto.",
            }
        except Exception as exc:
            return {"error": f"Error al eliminar objeto astronomico: {exc}"}

    async def listar_planetas_habitables(self, puntaje_minimo: float = 0.7) -> Dict[str, Any]:
        try:
            puntaje_minimo = float(puntaje_minimo)
            planetas = await self.repo_objetos.listar_planetas_por_habitabilidad(puntaje_minimo)
            return {"puntaje_minimo": puntaje_minimo, "planetas": [dump_model(p) for p in planetas]}
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
            id_objeto = to_optional_int(id_objeto, "id_objeto")
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
            embeddings = await self._guardar_embeddings_documento(
                documento.id_doc,
                contenido_texto,
                estrategia_chunking,
            )
            return {
                "documento": dump_model(documento),
                "embeddings": embeddings,
                "chunks_generados": len(embeddings),
                "estrategia_chunking": estrategia_chunking,
            }
        except Exception as exc:
            return {"error": f"Error al crear documento con embeddings: {exc}"}

    async def crear_imagen_con_embedding(
        self,
        ruta_archivo: str = "",
        descripcion: Optional[str] = None,
        etiquetas: Optional[List[str]] = None,
        id_doc: Optional[int] = None,
        imagen_base64: Optional[str] = None,
        extension: str = "png",
    ) -> Dict[str, Any]:
        try:
            id_doc = to_optional_int(id_doc, "id_doc")
            if not ruta_archivo and not imagen_base64:
                return {"error": "Debe proporcionar ruta_archivo o imagen_base64."}
            ruta_a_registrar = ruta_archivo
            if imagen_base64:
                ruta_a_registrar = str(self._guardar_imagen_base64(imagen_base64, extension))

            imagen = await self.repo_documentos.crear_imagen(
                Imagen(
                    id_imagen=-1,
                    ruta_archivo=ruta_a_registrar,
                    descripcion=descripcion,
                    etiquetas=etiquetas,
                    id_doc=id_doc,
                )
            )
            vector = await self.codificador_imagen.codificar_imagen(ruta_a_registrar)
            modelo = await self.codificador_imagen.nombre_modelo()
            id_embedding = await self.repo_documentos.guardar_embedding_imagen(
                imagen.id_imagen,
                vector,
                modelo,
            )
            return {
                "imagen": dump_model(imagen),
                "embedding_generado": True,
                "id_embedding": id_embedding,
                "modelo_embedding": modelo,
            }
        except Exception as exc:
            return {"error": f"Error al crear imagen: {exc}"}

    async def reemplazar_imagen_con_embedding(
        self,
        id_imagen: int,
        ruta_archivo: str = "",
        descripcion: Optional[str] = None,
        etiquetas: Optional[List[str]] = None,
        id_doc: Optional[int] = None,
        imagen_base64: Optional[str] = None,
        extension: str = "png",
    ) -> Dict[str, Any]:
        try:
            id_imagen = to_int(id_imagen, "id_imagen")
            id_doc = to_optional_int(id_doc, "id_doc")
            if not ruta_archivo and not imagen_base64:
                return {"error": "Debe proporcionar ruta_archivo o imagen_base64."}

            ruta_a_registrar = ruta_archivo
            if imagen_base64:
                ruta_a_registrar = str(self._guardar_imagen_base64(imagen_base64, extension))

            imagen = await self.repo_documentos.actualizar_imagen(
                Imagen(
                    id_imagen=id_imagen,
                    ruta_archivo=ruta_a_registrar,
                    descripcion=descripcion,
                    etiquetas=etiquetas,
                    id_doc=id_doc,
                )
            )
            embeddings_eliminados = await self.repo_documentos.eliminar_embeddings_imagen(id_imagen)
            vector = await self.codificador_imagen.codificar_imagen(ruta_a_registrar)
            modelo = await self.codificador_imagen.nombre_modelo()
            id_embedding = await self.repo_documentos.guardar_embedding_imagen(
                id_imagen,
                vector,
                modelo,
            )
            return {
                "imagen": dump_model(imagen),
                "embedding_generado": True,
                "id_embedding": id_embedding,
                "embeddings_eliminados": embeddings_eliminados,
                "modelo_embedding": modelo,
            }
        except Exception as exc:
            return {"error": f"Error al reemplazar imagen: {exc}"}

    async def eliminar_imagen_astronomica(self, id_imagen: int) -> Dict[str, Any]:
        try:
            id_imagen = to_int(id_imagen, "id_imagen")
            if id_imagen <= 0:
                return {"error": "id_imagen debe ser un entero positivo."}

            eliminada = await self.repo_documentos.eliminar_imagen(id_imagen)
            if not eliminada:
                return {"error": "No existe una imagen con ese id.", "id_imagen": id_imagen}
            return {
                "id_imagen": id_imagen,
                "eliminada": True,
                "confirmacion": "Imagen eliminada correctamente. Sus embeddings asociados se eliminaron por cascada.",
            }
        except Exception as exc:
            return {"error": f"Error al eliminar imagen: {exc}"}

    def _guardar_imagen_base64(self, imagen_base64: str, extension: str) -> Path:
        contenido = imagen_base64.strip()
        if "," in contenido and contenido.lower().startswith("data:"):
            contenido = contenido.split(",", 1)[1]

        extension_limpia = (extension or "png").strip().lower().lstrip(".")
        directorio = Path(__file__).resolve().parent.parent / "database" / "imagenes"
        directorio.mkdir(parents=True, exist_ok=True)

        ruta = directorio / f"{uuid.uuid4().hex}.{extension_limpia}"
        ruta.write_bytes(base64.b64decode(contenido))
        return ruta

    async def generar_embeddings_imagenes_pendientes(self, limite: int = 50) -> Dict[str, Any]:
        try:
            limite = int(limite)
            if limite <= 0:
                return {"error": "limite debe ser un entero positivo."}

            imagenes = await self.repo_documentos.listar_imagenes_sin_embedding(limite)
            modelo = await self.codificador_imagen.nombre_modelo()
            generadas = []
            errores = []

            for imagen in imagenes:
                try:
                    vector = await self.codificador_imagen.codificar_imagen(imagen.ruta_archivo)
                    id_embedding = await self.repo_documentos.guardar_embedding_imagen(
                        imagen.id_imagen,
                        vector,
                        modelo,
                    )
                    generadas.append({
                        "id_imagen": imagen.id_imagen,
                        "ruta_archivo": imagen.ruta_archivo,
                        "id_embedding": id_embedding,
                    })
                except Exception as exc:
                    errores.append({
                        "id_imagen": imagen.id_imagen,
                        "ruta_archivo": imagen.ruta_archivo,
                        "error": str(exc),
                    })

            return {
                "pendientes_encontradas": len(imagenes),
                "embeddings_generados": generadas,
                "errores": errores,
                "total_generados": len(generadas),
                "total_errores": len(errores),
                "modelo_embedding": modelo,
            }
        except Exception as exc:
            return {"error": f"Error al generar embeddings pendientes: {exc}"}

    async def crear_telescopio(self, nombre: str, tipo: Optional[str] = None, ubicacion: Optional[str] = None) -> Dict[str, Any]:
        try:
            telescopio = await self.repo_observaciones.crear_telescopio(
                nombre=nombre,
                tipo=tipo,
                ubicacion=ubicacion,
            )
            return {"telescopio": dump_model(telescopio)}
        except Exception as exc:
            return {"error": f"Error al crear telescopio: {exc}"}

    async def obtener_telescopio(self, id_telescopio: int) -> Dict[str, Any]:
        try:
            id_telescopio = to_int(id_telescopio, "id_telescopio")
            telescopio = await self.repo_observaciones.obtener_telescopio_por_id(id_telescopio)
            return {"telescopio": dump_model(telescopio)} if telescopio else {"error": "Telescopio no encontrado."}
        except Exception as exc:
            return {"error": f"Error al obtener telescopio: {exc}"}

    async def listar_telescopios(self) -> Dict[str, Any]:
        try:
            telescopios = await self.repo_observaciones.listar_telescopios()
            return {"telescopios": [dump_model(t) for t in telescopios]}
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
            id_telescopio = to_int(id_telescopio, "id_telescopio")
            id_objeto = to_int(id_objeto, "id_objeto")
            fecha_obs = date.fromisoformat(fecha) if isinstance(fecha, str) else (fecha or date.today())
            observacion = await self.repo_observaciones.crear_observacion(
                id_telescopio=id_telescopio,
                id_objeto=id_objeto,
                fecha=fecha_obs,
                descripcion=descripcion,
            )
            return {"observacion": dump_model(observacion)}
        except Exception as exc:
            return {"error": f"Error al crear observacion: {exc}"}

    async def listar_observaciones_por_objeto(self, id_objeto: int) -> Dict[str, Any]:
        try:
            obs = await self.repo_observaciones.listar_observaciones_por_objeto(to_int(id_objeto, "id_objeto"))
            return {"observaciones": [dump_model(o) for o in obs]}
        except Exception as exc:
            return {"error": f"Error al listar observaciones por objeto: {exc}"}

    async def listar_observaciones_por_telescopio(self, id_telescopio: int) -> Dict[str, Any]:
        try:
            obs = await self.repo_observaciones.listar_observaciones_por_telescopio(to_int(id_telescopio, "id_telescopio"))
            return {"observaciones": [dump_model(o) for o in obs]}
        except Exception as exc:
            return {"error": f"Error al listar observaciones por telescopio: {exc}"}

    async def _crear_documento_embedding_descripcion(self, id_objeto: int, titulo: str, contenido: str) -> int:
        documento = await self.repo_documentos.crear_documento(
            Documento(
                id_doc=-1,
                titulo=titulo,
                idioma="es",
                fecha=date.today(),
                fuente="Objeto_Astronomico.descripcion_cientifica",
                contenido_texto=contenido,
                id_objeto=id_objeto,
            )
        )
        embeddings = await self._guardar_embeddings_documento(documento.id_doc, contenido, "sentence")
        return embeddings[0] if embeddings else None

    async def _guardar_embeddings_documento(
        self,
        id_doc: int,
        contenido: str,
        estrategia_chunking: str,
    ) -> List[int]:
        modelo = await self.codificador.nombre_modelo()
        embeddings = []
        for chunk in self.chunking.dividir(contenido, estrategia_chunking):
            vector = await self.codificador.codificar_texto(chunk.contenido)
            id_embedding = await self.repo_documentos.guardar_embedding_texto(
                id_doc,
                chunk.chunk_id,
                vector,
                modelo,
                estrategia_chunking,
                chunk.contenido,
            )
            embeddings.append(id_embedding)
        return embeddings
