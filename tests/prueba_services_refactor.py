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
