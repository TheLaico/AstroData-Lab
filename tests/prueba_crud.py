"""
Pruebas unitarias para tools/gestion_objetos.py (GestionObjetos)
y database/repositorio_objetos.py de AstroData Lab.

Ejecutar con:
    pytest tests/prueba_crud.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vector_ejemplo() -> List[float]:
    """Fixture con vector de 384 dimensiones para embeddings."""
    import random
    random.seed(7)
    return [random.random() for _ in range(384)]


@pytest.fixture
def mock_codificador(vector_ejemplo):
    """Fixture con codificador mockeado que retorna vector de ejemplo."""
    from embeddings.interfaz_codificador import CodificadorBase
    codificador = MagicMock(spec=CodificadorBase)
    codificador.codificar_texto = AsyncMock(return_value=vector_ejemplo)
    codificador.nombre_modelo = AsyncMock(return_value="all-MiniLM-L6-v2")
    return codificador


@pytest.fixture
def objeto_mock():
    """Fixture con ObjetoAstronomico simulado retornado por el repositorio."""
    objeto = MagicMock()
    objeto.id_objeto = 42
    objeto.nombre = "Kepler-452b"
    objeto.descripcion_cientifica = "Exoplaneta superterrestre en órbita habitable"
    return objeto


@pytest.fixture
def mock_repo_objetos(objeto_mock):
    """Fixture con RepositorioObjetos mockeado."""
    from database.repositorio_objetos import RepositorioObjetos
    repo = MagicMock(spec=RepositorioObjetos)
    repo.crear_objeto = AsyncMock(return_value=objeto_mock)
    repo.obtener_objeto_por_id = AsyncMock(return_value=objeto_mock)
    repo.actualizar_descripcion = AsyncMock(return_value=objeto_mock)
    repo.eliminar_objeto = AsyncMock(return_value=True)
    repo.listar_planetas_por_habitabilidad = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_repo_documentos():
    """Fixture con RepositorioDocumentos mockeado."""
    from database.repositorio_documentos import RepositorioDocumentos
    repo = MagicMock(spec=RepositorioDocumentos)
    repo.guardar_embedding_texto = AsyncMock(return_value=1)
    return repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crear_planeta_exitoso(mock_codificador, mock_repo_objetos, mock_repo_documentos):
    """Verifica que crear_objeto_astronomico con tipo 'planeta' retorne
    un dict con id_objeto asignado y los datos del objeto creado."""
    with patch("tools.gestion_objetos.RepositorioObjetos", return_value=mock_repo_objetos), \
         patch("tools.gestion_objetos.RepositorioDocumentos", return_value=mock_repo_documentos):

        from tools.gestion_objetos import GestionObjetos
        gestion = GestionObjetos(codificador=mock_codificador)
        resultado = await gestion.crear_objeto_astronomico(
            nombre="Kepler-452b",
            tipo="planeta",
            descripcion_cientifica="Exoplaneta superterrestre en órbita habitable",
            atributos={"id_tipo_planeta": 1, "masa": 5.0, "temperatura": 265, "id_sistema": 3}
        )

    assert "error" not in resultado
    assert "id_objeto" in resultado
    assert resultado["id_objeto"] == 42
    assert resultado["tipo"] == "planeta"


@pytest.mark.asyncio
async def test_crear_objeto_tipo_invalido(mock_codificador, mock_repo_objetos, mock_repo_documentos):
    """Verifica que crear_objeto_astronomico retorne un error descriptivo
    en español cuando el tipo no pertenece a los tipos válidos del sistema."""
    with patch("tools.gestion_objetos.RepositorioObjetos", return_value=mock_repo_objetos), \
         patch("tools.gestion_objetos.RepositorioDocumentos", return_value=mock_repo_documentos):

        from tools.gestion_objetos import GestionObjetos
        gestion = GestionObjetos(codificador=mock_codificador)
        resultado = await gestion.crear_objeto_astronomico(
            nombre="Objeto Raro",
            tipo="asteroide",
            descripcion_cientifica="Cuerpo menor del sistema solar",
            atributos={}
        )

    assert "error" in resultado
    assert len(resultado["error"]) > 5


@pytest.mark.asyncio
async def test_crear_objeto_genera_embedding(
    mock_codificador, mock_repo_objetos, mock_repo_documentos
):
    """Verifica que al crear un objeto astronómico, se llame al codificador
    para generar el embedding de descripcion_cientifica exactamente una vez."""
    with patch("tools.gestion_objetos.RepositorioObjetos", return_value=mock_repo_objetos), \
         patch("tools.gestion_objetos.RepositorioDocumentos", return_value=mock_repo_documentos):

        from tools.gestion_objetos import GestionObjetos
        gestion = GestionObjetos(codificador=mock_codificador)
        await gestion.crear_objeto_astronomico(
            nombre="Vega",
            tipo="estrella",
            descripcion_cientifica="Estrella tipo A0 a 25 años luz de distancia",
            atributos={"id_tipo_estrella": 2, "masa": 2.1, "temperatura": 9600, "id_sistema": 5}
        )

    mock_codificador.codificar_texto.assert_called_once()


@pytest.mark.asyncio
async def test_actualizar_descripcion_regenera_embedding(
    mock_codificador, mock_repo_objetos, mock_repo_documentos
):
    """Verifica que al actualizar descripcion_cientifica, se regenere el
    embedding automáticamente llamando al codificador una vez."""
    with patch("tools.gestion_objetos.RepositorioObjetos", return_value=mock_repo_objetos), \
         patch("tools.gestion_objetos.RepositorioDocumentos", return_value=mock_repo_documentos):

        from tools.gestion_objetos import GestionObjetos
        gestion = GestionObjetos(codificador=mock_codificador)
        resultado = await gestion.actualizar_objeto_astronomico(
            id_objeto=42,
            campos={"descripcion_cientifica": "Nueva descripción científica actualizada"}
        )

    assert "error" not in resultado
    assert resultado.get("embedding_regenerado") is True
    mock_codificador.codificar_texto.assert_called_once()


@pytest.mark.asyncio
async def test_actualizar_campo_no_descripcion_no_regenera(
    mock_codificador, mock_repo_objetos, mock_repo_documentos
):
    """Verifica que actualizar un campo diferente a descripcion_cientifica
    NO regenere el embedding, es decir, el codificador no debe ser llamado."""
    with patch("tools.gestion_objetos.RepositorioObjetos", return_value=mock_repo_objetos), \
         patch("tools.gestion_objetos.RepositorioDocumentos", return_value=mock_repo_documentos):

        from tools.gestion_objetos import GestionObjetos
        gestion = GestionObjetos(codificador=mock_codificador)
        resultado = await gestion.actualizar_objeto_astronomico(
            id_objeto=42,
            campos={"nombre": "Nuevo Nombre"}
        )

    mock_codificador.codificar_texto.assert_not_called()
    assert resultado.get("embedding_regenerado") is False


@pytest.mark.asyncio
async def test_eliminar_objeto_exitoso(mock_codificador, mock_repo_objetos, mock_repo_documentos):
    """Verifica que eliminar_objeto_astronomico retorne confirmación con
    eliminado=True y llame al repositorio con el id correcto."""
    with patch("tools.gestion_objetos.RepositorioObjetos", return_value=mock_repo_objetos), \
         patch("tools.gestion_objetos.RepositorioDocumentos", return_value=mock_repo_documentos):

        from tools.gestion_objetos import GestionObjetos
        gestion = GestionObjetos(codificador=mock_codificador)
        resultado = await gestion.eliminar_objeto_astronomico(id_objeto=42)

    assert resultado.get("eliminado") is True
    assert "confirmacion" in resultado
    mock_repo_objetos.eliminar_objeto.assert_called_once_with(42)


@pytest.mark.asyncio
async def test_eliminar_objeto_inexistente(
    mock_codificador, mock_repo_objetos, mock_repo_documentos
):
    """Verifica que eliminar_objeto_astronomico retorne un error descriptivo
    en español cuando el id no existe en la base de datos."""
    mock_repo_objetos.obtener_objeto_por_id = AsyncMock(return_value=None)

    with patch("tools.gestion_objetos.RepositorioObjetos", return_value=mock_repo_ob