"""
Pruebas unitarias para tools/consulta_rag.py (ToolsConsultaRAG)
y database/repositorio_documentos.py de AstroData Lab.

Ejecutar con:
    pytest tests/prueba_rag.py -v
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vector_ejemplo() -> List[float]:
    """Fixture con un vector de embeddings de 384 dimensiones (MiniLM)."""
    import random
    random.seed(42)
    return [random.random() for _ in range(384)]


@pytest.fixture
def consulta_ejemplo() -> dict:
    """Fixture con una consulta RAG de ejemplo reutilizable."""
    return {
        "texto_pregunta": "¿Cuáles son las características de Kepler-442b?",
        "top_k": 3,
    }


@pytest.fixture
def chunks_ejemplo(vector_ejemplo) -> List[dict]:
    """Fixture con lista de chunks simulados devueltos por el repositorio."""
    return [
        {"titulo": "Exoplanetas tipo K", "chunk_id": 1,
         "estrategia_chunking": "sentence", "similitud": 0.92},
        {"titulo": "Zona habitable estelar", "chunk_id": 2,
         "estrategia_chunking": "sentence", "similitud": 0.87},
        {"titulo": "Composición atmosférica estimada", "chunk_id": 3,
         "estrategia_chunking": "sentence", "similitud": 0.81},
    ]


@pytest.fixture
def mock_codificador(vector_ejemplo):
    """Fixture con codificador mockeado que retorna el vector de ejemplo."""
    from embeddings.interfaz_codificador import CodificadorBase
    codificador = MagicMock(spec=CodificadorBase)
    codificador.codificar_texto = AsyncMock(return_value=vector_ejemplo)
    codificador.nombre_modelo = AsyncMock(return_value="all-MiniLM-L6-v2")
    return codificador


@pytest.fixture
def mock_repo_consultas():
    """Fixture con repositorio de consultas mockeado."""
    from database.repositorio_consultas import RepositorioConsultas
    repo = MagicMock(spec=RepositorioConsultas)
    consulta_mock = MagicMock()
    consulta_mock.id_consulta = 1
    consulta_mock.texto_pregunta = "¿Cuáles son las características de Kepler-442b?"
    from datetime import datetime
    consulta_mock.fecha = datetime.now()
    repo.registrar_consulta = AsyncMock(return_value=consulta_mock)
    repo.guardar_embedding_consulta = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_repo_documentos(chunks_ejemplo):
    """Fixture con repositorio de documentos mockeado."""
    from database.repositorio_documentos import RepositorioDocumentos
    repo = MagicMock(spec=RepositorioDocumentos)
    repo.buscar_chunks_similares = AsyncMock(return_value=chunks_ejemplo)
    return repo


@pytest.fixture
def mock_repo_objetos():
    """Fixture con repositorio de objetos mockeado."""
    from database.repositorio_objetos import RepositorioObjetos
    repo = MagicMock(spec=RepositorioObjetos)
    repo.obtener_objeto_por_id = AsyncMock(return_value=None)
    repo.obtener_objeto_por_nombre = AsyncMock(return_value=None)
    repo.listar_documentos_por_objeto = AsyncMock(return_value=[])
    return repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_query_retorna_chunks(
    mock_codificador, mock_repo_consultas, mock_repo_documentos,
    mock_repo_objetos, consulta_ejemplo
):
    """Verifica que rag_query retorne al menos top_k chunks con los campos
    titulo, chunk_id y similitud en cada elemento de chunks_recuperados."""
    with patch("tools.consulta_rag.RepositorioConsultas", return_value=mock_repo_consultas), \
         patch("tools.consulta_rag.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.consulta_rag.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.consulta_rag import ToolsConsultaRAG
        tools = ToolsConsultaRAG(codificador=mock_codificador)
        resultado = await tools.rag_query(
            texto_pregunta=consulta_ejemplo["texto_pregunta"],
            top_k=consulta_ejemplo["top_k"]
        )

    assert "error" not in resultado
    chunks = resultado["chunks_recuperados"]
    assert len(chunks) >= consulta_ejemplo["top_k"]
    for chunk in chunks:
        assert "titulo" in chunk
        assert "chunk_id" in chunk
        assert "similitud" in chunk


@pytest.mark.asyncio
async def test_rag_query_registra_consulta(
    mock_codificador, mock_repo_consultas, mock_repo_documentos,
    mock_repo_objetos, consulta_ejemplo
):
    """Verifica que rag_query llame a registrar_consulta() exactamente una vez."""
    with patch("tools.consulta_rag.RepositorioConsultas", return_value=mock_repo_consultas), \
         patch("tools.consulta_rag.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.consulta_rag.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.consulta_rag import ToolsConsultaRAG
        tools = ToolsConsultaRAG(codificador=mock_codificador)
        await tools.rag_query(texto_pregunta=consulta_ejemplo["texto_pregunta"])

    mock_repo_consultas.registrar_consulta.assert_called_once()


@pytest.mark.asyncio
async def test_rag_query_guarda_embedding(
    mock_codificador, mock_repo_consultas, mock_repo_documentos,
    mock_repo_objetos, consulta_ejemplo, vector_ejemplo
):
    """Verifica que rag_query llame a guardar_embedding_consulta()
    pasando el vector correcto generado por el codificador."""
    with patch("tools.consulta_rag.RepositorioConsultas", return_value=mock_repo_consultas), \
         patch("tools.consulta_rag.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.consulta_rag.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.consulta_rag import ToolsConsultaRAG
        tools = ToolsConsultaRAG(codificador=mock_codificador)
        await tools.rag_query(texto_pregunta=consulta_ejemplo["texto_pregunta"])

    mock_repo_consultas.guardar_embedding_consulta.assert_called_once()
    args, kwargs = mock_repo_consultas.guardar_embedding_consulta.call_args
    vector_usado = kwargs.get("vector") or args[1]
    assert vector_usado == vector_ejemplo


@pytest.mark.asyncio
async def test_rag_query_texto_vacio(
    mock_codificador, mock_repo_consultas, mock_repo_documentos, mock_repo_objetos
):
    """Verifica que rag_query retorne un error descriptivo en español
    cuando texto_pregunta está vacío o contiene solo espacios."""
    with patch("tools.consulta_rag.RepositorioConsultas", return_value=mock_repo_consultas), \
         patch("tools.consulta_rag.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.consulta_rag.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.consulta_rag import ToolsConsultaRAG
        tools = ToolsConsultaRAG(codificador=mock_codificador)

        resultado_vacio = await tools.rag_query(texto_pregunta="")
        resultado_espacios = await tools.rag_query(texto_pregunta="   ")

    assert "error" in resultado_vacio
    assert len(resultado_vacio["error"]) > 5
    assert "error" in resultado_espacios


@pytest.mark.asyncio
async def test_obtener_contexto_sin_id_ni_nombre(
    mock_codificador, mock_repo_consultas, mock_repo_documentos, mock_repo_objetos
):
    """Verifica que obtener_contexto_objeto retorne un error descriptivo
    cuando no se provee ni id_objeto ni nombre."""
    with patch("tools.consulta_rag.RepositorioConsultas", return_value=mock_repo_consultas), \
         patch("tools.consulta_rag.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.consulta_rag.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.consulta_rag import ToolsConsultaRAG
        tools = ToolsConsultaRAG(codificador=mock_codificador)
        resultado = await tools.obtener_contexto_objeto(id_objeto=None, nombre=None)

    assert "error" in resultado
    assert len(resultado["error"]) > 5


@pytest.mark.asyncio
async def test_obtener_contexto_planeta_incluye_habitabilidad(
    mock_codificador, mock_repo_consultas, mock_repo_documentos, mock_repo_objetos
):
    """Verifica que cuando se consulta contexto de un planeta, el retorno
    incluye la clave evaluaciones_habitabilidad (puede estar vacía si no hay datos)."""
    objeto_mock = MagicMock()
    objeto_mock.id_objeto = 7
    objeto_mock.nombre = "Kepler-442b"
    objeto_mock.descripcion_cientifica = "Exoplaneta superterrestre en zona habitable"

    mock_repo_objetos.obtener_objeto_por_id = AsyncMock(return_value=objeto_mock)
    mock_repo_objetos.listar_documentos_por_objeto = AsyncMock(return_value=[])
    mock_repo_objetos.obtener_caracteristicas_ambientales = AsyncMock(return_value=[])

    with patch("tools.consulta_rag.RepositorioConsultas", return_value=mock_repo_consultas), \
         patch("tools.consulta_rag.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.consulta_rag.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.consulta_rag import ToolsConsultaRAG
        tools = ToolsConsultaRAG(codificador=mock_codificador)
        resultado = await tools.obtener_contexto_objeto(id_objeto=7)

    assert "error" not in resultado
    assert "objeto" in resultado
    assert "caracteristicas_ambientales" in resultado