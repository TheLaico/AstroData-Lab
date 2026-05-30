from unittest.mock import AsyncMock, MagicMock

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
