from unittest.mock import AsyncMock, MagicMock
import base64
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_rag_service_orquesta_repositorios():
    from services.rag_service import ConsultaRAGService

    codificador = MagicMock()
    codificador.codificar_texto = AsyncMock(return_value=[0.1] * 384)
    codificador.nombre_modelo = AsyncMock(return_value="all-MiniLM-L6-v2")

    consulta = MagicMock(id_consulta=10)
    repo_consultas = MagicMock()
    repo_consultas.registrar_consulta = AsyncMock(return_value=consulta)
    repo_consultas.guardar_embedding_consulta = AsyncMock(return_value=1)

    repo_documentos = MagicMock()
    repo_documentos.buscar_chunks_similares = AsyncMock(return_value=[
        {"id_doc": 7, "titulo": "Habitabilidad", "chunk_id": 0, "similitud": 0.9, "contenido": "agua liquida"}
    ])

    service = ConsultaRAGService(codificador, repo_consultas, repo_documentos, MagicMock())
    resultado = await service.rag_query("Que hace habitable a la Tierra?", id_usuario=1)

    assert resultado["id_consulta"] == 10
    assert resultado["chunks_recuperados"][0]["titulo"] == "Habitabilidad"
    repo_consultas.registrar_consulta.assert_called_once()
    repo_consultas.guardar_embedding_consulta.assert_called_once()


@pytest.mark.asyncio
async def test_gestion_service_convierte_ids_string():
    from services.objetos_service import GestionObjetosService

    repo_observaciones = MagicMock()
    repo_observaciones.listar_observaciones_por_objeto = AsyncMock(return_value=[])

    service = GestionObjetosService(
        codificador=MagicMock(),
        repo_objetos=MagicMock(),
        repo_documentos=MagicMock(),
        repo_observaciones=repo_observaciones,
    )

    resultado = await service.listar_observaciones_por_objeto("4")

    assert resultado == {"observaciones": []}
    repo_observaciones.listar_observaciones_por_objeto.assert_called_once_with(4)


@pytest.mark.asyncio
async def test_gestion_service_chunking_genera_multiples_embeddings():
    from models.documento_model import Documento
    from services.objetos_service import GestionObjetosService

    codificador = MagicMock()
    codificador.codificar_texto = AsyncMock(return_value=[0.2] * 384)
    codificador.nombre_modelo = AsyncMock(return_value="all-MiniLM-L6-v2")

    repo_documentos = MagicMock()
    repo_documentos.crear_documento = AsyncMock(
        return_value=Documento(
            id_doc=99,
            titulo="Doc largo",
            idioma="es",
            fecha=None,
            fuente=None,
            contenido_texto="x",
            id_objeto=4,
        )
    )
    repo_documentos.guardar_embedding_texto = AsyncMock(side_effect=[10, 11])

    service = GestionObjetosService(
        codificador=codificador,
        repo_objetos=MagicMock(),
        repo_documentos=repo_documentos,
        repo_observaciones=MagicMock(),
    )
    contenido = " ".join(f"palabra{i}" for i in range(210))

    resultado = await service.crear_documento_con_embeddings(
        titulo="Doc largo",
        contenido_texto=contenido,
        estrategia_chunking="fixed",
    )

    assert resultado["embeddings"] == [10, 11]
    assert resultado["chunks_generados"] == 2
    assert repo_documentos.guardar_embedding_texto.call_count == 2


@pytest.mark.asyncio
async def test_gestion_service_crear_imagen_genera_embedding_clip():
    from models.imagen_model import Imagen
    from services.objetos_service import GestionObjetosService

    codificador_imagen = MagicMock()
    codificador_imagen.codificar_imagen = AsyncMock(return_value=[0.5] * 512)
    codificador_imagen.nombre_modelo = AsyncMock(return_value="openai/clip-vit-base-patch32")

    repo_documentos = MagicMock()
    repo_documentos.crear_imagen = AsyncMock(
        return_value=Imagen(
            id_imagen=12,
            ruta_archivo="/imgs/saturno.jpg",
            descripcion="Planeta con anillos",
            etiquetas=["planeta", "anillos"],
            id_doc=3,
        )
    )
    repo_documentos.guardar_embedding_imagen = AsyncMock(return_value=44)

    service = GestionObjetosService(
        codificador=MagicMock(),
        codificador_imagen=codificador_imagen,
        repo_objetos=MagicMock(),
        repo_documentos=repo_documentos,
        repo_observaciones=MagicMock(),
    )

    resultado = await service.crear_imagen_con_embedding(
        ruta_archivo="/imgs/saturno.jpg",
        descripcion="Planeta con anillos",
        etiquetas=["planeta", "anillos"],
        id_doc=3,
    )

    assert resultado["embedding_generado"] is True
    assert resultado["id_embedding"] == 44
    repo_documentos.guardar_embedding_imagen.assert_called_once_with(
        12,
        [0.5] * 512,
        "openai/clip-vit-base-patch32",
    )


@pytest.mark.asyncio
async def test_gestion_service_crear_imagen_desde_base64():
    from models.imagen_model import Imagen
    from services.objetos_service import GestionObjetosService

    codificador_imagen = MagicMock()
    codificador_imagen.codificar_imagen = AsyncMock(return_value=[0.7] * 512)
    codificador_imagen.nombre_modelo = AsyncMock(return_value="openai/clip-vit-base-patch32")

    rutas_creadas = []

    async def crear_imagen(datos):
        rutas_creadas.append(Path(datos.ruta_archivo))
        return Imagen(
            id_imagen=21,
            ruta_archivo=datos.ruta_archivo,
            descripcion=datos.descripcion,
            etiquetas=datos.etiquetas,
            id_doc=datos.id_doc,
        )

    repo_documentos = MagicMock()
    repo_documentos.crear_imagen = AsyncMock(side_effect=crear_imagen)
    repo_documentos.guardar_embedding_imagen = AsyncMock(return_value=55)

    service = GestionObjetosService(
        codificador=MagicMock(),
        codificador_imagen=codificador_imagen,
        repo_objetos=MagicMock(),
        repo_documentos=repo_documentos,
        repo_observaciones=MagicMock(),
    )

    resultado = await service.crear_imagen_con_embedding(
        imagen_base64=base64.b64encode(b"imagen falsa").decode("ascii"),
        extension="png",
        descripcion="Jupiter adjunto",
        etiquetas=["jupiter"],
    )

    try:
        assert resultado["embedding_generado"] is True
        assert rutas_creadas
        assert rutas_creadas[0].exists()
        codificador_imagen.codificar_imagen.assert_called_once_with(str(rutas_creadas[0]))
    finally:
        for ruta in rutas_creadas:
            ruta.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_gestion_service_genera_embeddings_pendientes():
    from models.imagen_model import Imagen
    from services.objetos_service import GestionObjetosService

    codificador_imagen = MagicMock()
    codificador_imagen.codificar_imagen = AsyncMock(return_value=[0.6] * 512)
    codificador_imagen.nombre_modelo = AsyncMock(return_value="openai/clip-vit-base-patch32")

    repo_documentos = MagicMock()
    repo_documentos.listar_imagenes_sin_embedding = AsyncMock(return_value=[
        Imagen(
            id_imagen=2,
            ruta_archivo="https://example.com/jupiter.jpg",
            descripcion="Jupiter",
            etiquetas=None,
            id_doc=None,
        )
    ])
    repo_documentos.guardar_embedding_imagen = AsyncMock(return_value=88)

    service = GestionObjetosService(
        codificador=MagicMock(),
        codificador_imagen=codificador_imagen,
        repo_objetos=MagicMock(),
        repo_documentos=repo_documentos,
        repo_observaciones=MagicMock(),
    )

    resultado = await service.generar_embeddings_imagenes_pendientes(limite=10)

    assert resultado["total_generados"] == 1
    assert resultado["total_errores"] == 0
    repo_documentos.listar_imagenes_sin_embedding.assert_called_once_with(10)
    repo_documentos.guardar_embedding_imagen.assert_called_once_with(
        2,
        [0.6] * 512,
        "openai/clip-vit-base-patch32",
    )


@pytest.mark.asyncio
async def test_gestion_service_reemplaza_imagen_y_regenera_embedding():
    from models.imagen_model import Imagen
    from services.objetos_service import GestionObjetosService

    codificador_imagen = MagicMock()
    codificador_imagen.codificar_imagen = AsyncMock(return_value=[0.8] * 512)
    codificador_imagen.nombre_modelo = AsyncMock(return_value="openai/clip-vit-base-patch32")

    repo_documentos = MagicMock()
    repo_documentos.actualizar_imagen = AsyncMock(
        return_value=Imagen(
            id_imagen=9,
            ruta_archivo="https://example.com/jupiter.jpg",
            descripcion="Jupiter actualizado",
            etiquetas=["jupiter"],
            id_doc=9,
        )
    )
    repo_documentos.eliminar_embeddings_imagen = AsyncMock(return_value=1)
    repo_documentos.guardar_embedding_imagen = AsyncMock(return_value=99)

    service = GestionObjetosService(
        codificador=MagicMock(),
        codificador_imagen=codificador_imagen,
        repo_objetos=MagicMock(),
        repo_documentos=repo_documentos,
        repo_observaciones=MagicMock(),
    )

    resultado = await service.reemplazar_imagen_con_embedding(
        id_imagen="9",
        ruta_archivo="https://example.com/jupiter.jpg",
        descripcion="Jupiter actualizado",
        etiquetas=["jupiter"],
        id_doc="9",
    )

    assert resultado["embedding_generado"] is True
    assert resultado["id_embedding"] == 99
    assert resultado["embeddings_eliminados"] == 1
    repo_documentos.eliminar_embeddings_imagen.assert_called_once_with(9)
    repo_documentos.guardar_embedding_imagen.assert_called_once_with(
        9,
        [0.8] * 512,
        "openai/clip-vit-base-patch32",
    )


@pytest.mark.asyncio
async def test_gestion_service_elimina_imagen_astronomica():
    from services.objetos_service import GestionObjetosService

    repo_documentos = MagicMock()
    repo_documentos.eliminar_imagen = AsyncMock(return_value=True)

    service = GestionObjetosService(
        codificador=MagicMock(),
        repo_objetos=MagicMock(),
        repo_documentos=repo_documentos,
        repo_observaciones=MagicMock(),
    )

    resultado = await service.eliminar_imagen_astronomica("11")

    assert resultado["eliminada"] is True
    repo_documentos.eliminar_imagen.assert_called_once_with(11)


@pytest.mark.asyncio
async def test_eliminar_objeto_informa_politica_documentos():
    from services.objetos_service import GestionObjetosService

    repo_objetos = MagicMock()
    repo_objetos.obtener_objeto_por_id = AsyncMock(return_value=MagicMock(id_objeto=4))
    repo_objetos.eliminar_objeto = AsyncMock(return_value=True)

    service = GestionObjetosService(
        codificador=MagicMock(),
        repo_objetos=repo_objetos,
        repo_documentos=MagicMock(),
        repo_observaciones=MagicMock(),
    )

    resultado = await service.eliminar_objeto_astronomico(4)

    assert resultado["eliminado"] is True
    assert "documentos asociados se conservan" in resultado["politica_documentos"]
