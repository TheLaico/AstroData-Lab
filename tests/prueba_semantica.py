"""
Pruebas unitarias para tools/busqueda_semantica.py (BusquedaSematica)
y database/repositorio_documentos.py de AstroData Lab.

Ejecutar con:
    pytest tests/prueba_semantica.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vector_ejemplo() -> List[float]:
    """Fixture con vector aleatorio de 384 dimensiones (MiniLM)."""
    import random
    random.seed(99)
    return [random.random() for _ in range(384)]


@pytest.fixture
def mock_codificador(vector_ejemplo):
    """Fixture con codificador de texto mockeado."""
    from database.embeddings.interfaz_codificador import CodificadorBase
    codificador = MagicMock(spec=CodificadorBase)
    codificador.codificar_texto = AsyncMock(return_value=vector_ejemplo)
    codificador.nombre_modelo = AsyncMock(return_value="all-MiniLM-L6-v2")
    return codificador


@pytest.fixture
def mock_codificador_imagen():
    """Fixture con codificador CLIP mockeado."""
    codificador = MagicMock()
    codificador.codificar_texto = AsyncMock(return_value=[0.3] * 512)
    codificador.codificar_imagen = AsyncMock(return_value=[0.4] * 512)
    codificador.nombre_modelo = AsyncMock(return_value="openai/clip-vit-base-patch32")
    return codificador


@pytest.fixture
def documentos_ejemplo() -> List[dict]:
    """Fixture con lista de documentos de prueba desordenados por similitud."""
    return [
        {"titulo": "Galaxias espirales", "chunk_id": 3,
         "estrategia_chunking": "sentence", "similitud": 0.60},
        {"titulo": "Habitabilidad exoplanetas", "chunk_id": 1,
         "estrategia_chunking": "semantic", "similitud": 0.91},
        {"titulo": "Estrellas tipo G", "chunk_id": 2,
         "estrategia_chunking": "fixed", "similitud": 0.75},
    ]


@pytest.fixture
def mock_repo_documentos(documentos_ejemplo):
    """Fixture con RepositorioDocumentos mockeado."""
    from database.repositorio_documentos import RepositorioDocumentos
    repo = MagicMock(spec=RepositorioDocumentos)
    repo.buscar_chunks_similares = AsyncMock(return_value=documentos_ejemplo)
    repo.buscar_imagenes_similares = AsyncMock(return_value=[
        {"id_imagen": 1, "ruta_archivo": "/imgs/nebulosa.jpg",
         "descripcion": "Nebulosa de Orión", "similitud": 0.88},
    ])
    return repo


@pytest.fixture
def mock_repo_objetos():
    """Fixture con RepositorioObjetos mockeado."""
    from database.repositorio_objetos import RepositorioObjetos
    repo = MagicMock(spec=RepositorioObjetos)
    objeto_mock = MagicMock()
    objeto_mock.id_objeto = 5
    objeto_mock.nombre = "Kepler-442b"
    objeto_mock.descripcion_cientifica = "Planeta en zona habitable"
    repo.obtener_objeto_por_id = AsyncMock(return_value=objeto_mock)
    return repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_busqueda_documentos_retorna_ordenados(
    mock_codificador, mock_repo_documentos, mock_repo_objetos
):
    """Verifica que los resultados de buscar_documentos_semanticos vengan
    ordenados de mayor a menor por puntuacion_similitud."""
    with patch("tools.busqueda_semantica.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.busqueda_semantica.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.busqueda_semantica import BusquedaSematica
        busqueda = BusquedaSematica(codificador=mock_codificador)
        resultado = await busqueda.buscar_documentos_semanticos(
            consulta="exoplanetas habitables", top_k=5
        )

    docs = resultado["documentos"]
    similitudes = [d["puntuacion_similitud"] for d in docs]
    assert similitudes == sorted(similitudes, reverse=True)


@pytest.mark.asyncio
async def test_busqueda_documentos_respeta_top_k(
    mock_codificador, mock_repo_documentos, mock_repo_objetos
):
    """Verifica que buscar_documentos_semanticos nunca retorne más
    documentos que el valor de top_k solicitado."""
    with patch("tools.busqueda_semantica.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.busqueda_semantica.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.busqueda_semantica import BusquedaSematica
        busqueda = BusquedaSematica(codificador=mock_codificador)
        resultado = await busqueda.buscar_documentos_semanticos(
            consulta="galaxias", top_k=2
        )

    assert len(resultado["documentos"]) <= 2


@pytest.mark.asyncio
async def test_busqueda_documentos_filtra_estrategia(
    mock_codificador, mock_repo_documentos, mock_repo_objetos
):
    """Verifica que el filtro por estrategia_chunking se pase correctamente
    al repositorio al invocar buscar_documentos_semanticos."""
    with patch("tools.busqueda_semantica.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.busqueda_semantica.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.busqueda_semantica import BusquedaSematica
        busqueda = BusquedaSematica(codificador=mock_codificador)
        await busqueda.buscar_documentos_semanticos(
            consulta="nebulosas", top_k=3, estrategia_chunking="semantic"
        )

    llamada = mock_repo_documentos.buscar_chunks_similares.call_args
    estrategia_pasada = llamada.kwargs.get("estrategia") or llamada.args[2]
    assert estrategia_pasada == "semantic"


@pytest.mark.asyncio
async def test_busqueda_imagenes_vectoriza_con_clip(
    mock_codificador, mock_codificador_imagen, mock_repo_documentos, mock_repo_objetos
):
    """Verifica que buscar_imagenes_semanticas use CodificadorTexto (no
    CodificadorImagen) para vectorizar la descripción de búsqueda."""
    with patch("tools.busqueda_semantica.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.busqueda_semantica.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.busqueda_semantica import BusquedaSematica
        busqueda = BusquedaSematica(
            codificador=mock_codificador,
            codificador_imagen=mock_codificador_imagen,
        )
        await busqueda.buscar_imagenes_semanticas(
            descripcion="Nebulosa roja con estrellas brillantes", top_k=3
        )

    # El codificador_texto debe haber sido usado para vectorizar
    mock_codificador_imagen.codificar_texto.assert_called_once()
    mock_codificador.codificar_texto.assert_not_called()


@pytest.mark.asyncio
async def test_obtener_info_objeto_por_imagen_retorna_respuesta_textual(
    mock_codificador, mock_codificador_imagen, mock_repo_documentos, mock_repo_objetos
):
    """Verifica que una busqueda por imagen pueda devolver informacion del objeto."""
    mock_repo_documentos.buscar_imagenes_similares = AsyncMock(return_value=[
        {
            "id_imagen": 7,
            "ruta_archivo": "/imgs/saturno.jpg",
            "descripcion": "Planeta con anillos",
            "etiquetas": "planeta,anillos",
            "id_doc": 3,
            "titulo_documento": "Saturno",
            "fuente_documento": "catalogo",
            "id_objeto": 9,
            "nombre_objeto": "Saturno",
            "tipo_objeto": "planeta",
            "descripcion_cientifica": "Planeta gigante gaseoso con sistema de anillos.",
            "similitud": 0.97,
        }
    ])

    with patch("tools.busqueda_semantica.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.busqueda_semantica.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.busqueda_semantica import BusquedaSematica
        busqueda = BusquedaSematica(
            codificador=mock_codificador,
            codificador_imagen=mock_codificador_imagen,
        )
        resultado = await busqueda.obtener_info_objeto_por_imagen(
            ruta_imagen="/tmp/consulta.jpg",
            top_k=1,
        )

    assert "Saturno" in resultado["respuesta_textual"]
    assert resultado["objeto_detectado"]["tipo"] == "planeta"
    mock_codificador_imagen.codificar_imagen.assert_called_once_with("/tmp/consulta.jpg")


@pytest.mark.asyncio
async def test_encontrar_planetas_excluye_referencia(
    mock_codificador, mock_repo_documentos, mock_repo_objetos
):
    """Verifica que el planeta de referencia no aparezca en los resultados
    de encontrar_planetas_similares."""
    # El repo devuelve chunks que incluyen el nombre del planeta de referencia
    mock_repo_documentos.buscar_chunks_similares = AsyncMock(return_value=[
        {"titulo": "Kepler-442b", "chunk_id": 5,
         "estrategia_chunking": "semantic", "similitud": 0.99},
        {"titulo": "Proxima Centauri b", "chunk_id": 6,
         "estrategia_chunking": "semantic", "similitud": 0.85},
    ])

    with patch("tools.busqueda_semantica.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.busqueda_semantica.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.busqueda_semantica import BusquedaSematica
        busqueda = BusquedaSematica(codificador=mock_codificador)
        resultado = await busqueda.encontrar_planetas_similares(id_planeta=5, top_k=5)

    nombres_resultado = [p["nombre"] for p in resultado["planetas_similares"]]
    assert "Kepler-442b" not in nombres_resultado


@pytest.mark.asyncio
async def test_encontrar_planetas_genera_embedding_si_no_existe(
    mock_codificador, mock_repo_documentos, mock_repo_objetos
):
    """Verifica que si el planeta de referencia no tiene embedding previo
    cacheado, se genere uno llamando al codificador antes de buscar."""
    with patch("tools.busqueda_semantica.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.busqueda_semantica.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.busqueda_semantica import BusquedaSematica
        busqueda = BusquedaSematica(codificador=mock_codificador)
        await busqueda.encontrar_planetas_similares(id_planeta=5, top_k=3)

    # Si no había embedding, debe haber llamado al codificador para generarlo
    mock_codificador.codificar_texto.assert_called()


@pytest.mark.asyncio
async def test_busqueda_query_vacia(
    mock_codificador, mock_repo_documentos, mock_repo_objetos
):
    """Verifica que buscar_documentos_semanticos retorne un error descriptivo
    en español cuando la consulta está vacía o solo tiene espacios en blanco."""
    with patch("tools.busqueda_semantica.RepositorioDocumentos", return_value=mock_repo_documentos), \
         patch("tools.busqueda_semantica.RepositorioObjetos", return_value=mock_repo_objetos):

        from tools.busqueda_semantica import BusquedaSematica
        busqueda = BusquedaSematica(codificador=mock_codificador)

        resultado_vacio = await busqueda.buscar_documentos_semanticos(consulta="")
        resultado_espacios = await busqueda.buscar_documentos_semanticos(consulta="   ")

    assert "error" in resultado_vacio
    assert len(resultado_vacio["error"]) > 5
    assert "error" in resultado_espacios
