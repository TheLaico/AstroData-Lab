"""
Pruebas unitarias para tools/evaluacion_ragas.py (ToolsEvaluacionRAGAS)
y database/repositorio_evaluaciones.py de AstroData Lab.

Ejecutar con:
    pytest tests/prueba_evaluacion.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, date


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluacion_ragas_mock():
    """Fixture con EvaluacionRAGAS de ejemplo con valores conocidos
    para validar los cálculos de promedio y calificación."""
    from models.evaluacion import EvaluacionRAGAS
    return EvaluacionRAGAS(
        id_evaluacion=1,
        faithfulness=0.90,
        answer_relevancy=0.85,
        context_recall=0.88,
        modelo_eval="claude-sonnet-4-20250514",
        fecha=datetime(2026, 3, 15, 10, 0, 0),
        id_consulta=42
    )


@pytest.fixture
def consulta_mock():
    """Fixture con Consulta simulada que existe en la BD."""
    consulta = MagicMock()
    consulta.id_consulta = 42
    consulta.texto_pregunta = "¿Qué es una nebulosa planetaria?"
    consulta.fecha = datetime.now()
    consulta.id_usuario = 1
    return consulta


@pytest.fixture
def mock_repo_evaluaciones(evaluacion_ragas_mock):
    """Fixture con RepositorioEvaluaciones mockeado."""
    from database.repositorio_evaluaciones import RepositorioEvaluaciones
    repo = MagicMock(spec=RepositorioEvaluaciones)
    repo.registrar_evaluacion_ragas = AsyncMock(return_value=evaluacion_ragas_mock)
    repo.listar_evaluaciones_por_usuario = AsyncMock(return_value=[evaluacion_ragas_mock])

    from models.evaluacion import ResumenEvaluacion
    resumen_mock = ResumenEvaluacion(evaluacion=evaluacion_ragas_mock)
    repo.calcular_resumen_usuario = AsyncMock(return_value=resumen_mock)
    return repo


@pytest.fixture
def mock_repo_consultas(consulta_mock):
    """Fixture con RepositorioConsultas mockeado."""
    from database.repositorio_consultas import RepositorioConsultas
    repo = MagicMock(spec=RepositorioConsultas)
    repo.obtener_consulta_por_id = AsyncMock(return_value=consulta_mock)
    return repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluar_respuesta_persiste_metricas(
    mock_repo_evaluaciones, mock_repo_consultas
):
    """Verifica que evaluar_respuesta_rag guarde las tres métricas
    (faithfulness, answer_relevancy, context_recall) en la BD llamando
    exactamente una vez a registrar_evaluacion_ragas."""
    with patch("tools.evaluacion_ragas.RepositorioEvaluaciones", return_value=mock_repo_evaluaciones), \
         patch("tools.evaluacion_ragas.RepositorioConsultas", return_value=mock_repo_consultas):

        from tools.evaluacion_ragas import ToolsEvaluacionRAGAS
        tools = ToolsEvaluacionRAGAS()
        await tools.evaluar_respuesta_rag(
            id_consulta=42,
            respuesta_generada="Una nebulosa planetaria es una nube de gas expulsada por una estrella.",
            contexto_recuperado=["Las nebulosas planetarias se forman al final de la vida de estrellas tipo solar."]
        )

    mock_repo_evaluaciones.registrar_evaluacion_ragas.assert_called_once()
    entrada_llamada = mock_repo_evaluaciones.registrar_evaluacion_ragas.call_args.args[0]
    assert hasattr(entrada_llamada, "faithfulness")
    assert hasattr(entrada_llamada, "answer_relevancy")
    assert hasattr(entrada_llamada, "context_recall")


@pytest.mark.asyncio
async def test_evaluar_respuesta_retorna_calificacion(
    mock_repo_evaluaciones, mock_repo_consultas
):
    """Verifica que el retorno de evaluar_respuesta_rag incluya el campo
    'calidad' con un valor válido: 'alta', 'media' o 'baja'."""
    with patch("tools.evaluacion_ragas.RepositorioEvaluaciones", return_value=mock_repo_evaluaciones), \
         patch("tools.evaluacion_ragas.RepositorioConsultas", return_value=mock_repo_consultas):

        from tools.evaluacion_ragas import ToolsEvaluacionRAGAS
        tools = ToolsEvaluacionRAGAS()
        resultado = await tools.evaluar_respuesta_rag(
            id_consulta=42,
            respuesta_generada="Respuesta sobre nebulosas.",
            contexto_recuperado=["Contexto sobre nebulosas planetarias."]
        )

    assert "error" not in resultado
    assert "calidad" in resultado
    assert resultado["calidad"] in ("alta", "media", "baja")


@pytest.mark.asyncio
async def test_calificacion_alta_cuando_promedio_mayor_08(
    mock_repo_consultas
):
    """Verifica que cuando el promedio de las tres métricas sea mayor a 0.8,
    la calificación retornada sea 'alta'."""
    from models.evaluacion import EvaluacionRAGAS
    eval_alta = EvaluacionRAGAS(
        id_evaluacion=2, faithfulness=0.85, answer_relevancy=0.90,
        context_recall=0.88, modelo_eval="test", fecha=datetime.now(), id_consulta=42
    )

    from database.repositorio_evaluaciones import RepositorioEvaluaciones
    repo_eval = MagicMock(spec=RepositorioEvaluaciones)
    repo_eval.registrar_evaluacion_ragas = AsyncMock(return_value=eval_alta)

    with patch("tools.evaluacion_ragas.RepositorioEvaluaciones", return_value=repo_eval), \
         patch("tools.evaluacion_ragas.RepositorioConsultas", return_value=mock_repo_consultas):

        from tools.evaluacion_ragas import ToolsEvaluacionRAGAS
        tools = ToolsEvaluacionRAGAS()
        # Usamos un texto con mucho solapamiento para forzar promedio alto
        resultado = await tools.evaluar_respuesta_rag(
            id_consulta=42,
            respuesta_generada="nebulosa planetaria estrella gas expulsada",
            contexto_recuperado=["nebulosa planetaria estrella gas expulsada forma vida solar"]
        )

    promedio = resultado["promedio_metricas"]
    assert promedio > 0.8
    assert resultado["calidad"] == "alta"


@pytest.mark.asyncio
async def test_calificacion_baja_cuando_promedio_menor_05(
    mock_repo_consultas
):
    """Verifica que cuando el promedio de las tres métricas sea menor a 0.5,
    la calificación retornada sea 'baja'."""
    from models.evaluacion import EvaluacionRAGAS
    eval_baja = EvaluacionRAGAS(
        id_evaluacion=3, faithfulness=0.10, answer_relevancy=0.15,
        context_recall=0.20, modelo_eval="test", fecha=datetime.now(), id_consulta=42
    )

    from database.repositorio_evaluaciones import RepositorioEvaluaciones
    repo_eval = MagicMock(spec=RepositorioEvaluaciones)
    repo_eval.registrar_evaluacion_ragas = AsyncMock(return_value=eval_baja)

    with patch("tools.evaluacion_ragas.RepositorioEvaluaciones", return_value=repo_eval), \
         patch("tools.evaluacion_ragas.RepositorioConsultas", return_value=mock_repo_consultas):

        from tools.evaluacion_ragas import ToolsEvaluacionRAGAS
        tools = ToolsEvaluacionRAGAS()
        resultado = await tools.evaluar_respuesta_rag(
            id_consulta=42,
            respuesta_generada="xyz abc completamente diferente e irrelevante",
            contexto_recuperado=["nebulosa planetaria estrella"]
        )

    promedio = resultado["promedio_metricas"]
    assert promedio < 0.5
    assert resultado["calidad"] == "baja"


@pytest.mark.asyncio
async def test_historial_convierte_fechas_string(
    mock_repo_evaluaciones, mock_repo_consultas
):
    """Verifica que las fechas en formato 'YYYY-MM-DD' se conviertan
    correctamente a objetos date antes de consultar el repositorio."""
    with patch("tools.evaluacion_ragas.RepositorioEvaluaciones", return_value=mock_repo_evaluaciones), \
         patch("tools.evaluacion_ragas.RepositorioConsultas", return_value=mock_repo_consultas):

        from tools.evaluacion_ragas import ToolsEvaluacionRAGAS
        tools = ToolsEvaluacionRAGAS()
        await tools.obtener_historial_evaluaciones(
            id_usuario=1,
            fecha_desde="2026-01-01",
            fecha_hasta="2026-05-31"
        )

    llamada = mock_repo_evaluaciones.listar_evaluaciones_por_usuario.call_args
    fecha_desde_pasada = llamada.kwargs.get("fecha_desde") or llamada.args[1]
    fecha_hasta_pasada = llamada.kwargs.get("fecha_hasta") or llamada.args[2]
    assert isinstance(fecha_desde_pasada, date)
    assert isinstance(fecha_hasta_pasada, date)
    assert fecha_desde_pasada == date(2026, 1, 1)
    assert fecha_hasta_pasada == date(2026, 5, 31)


@pytest.mark.asyncio
async def test_historial_incluye_resumen(
    mock_repo_evaluaciones, mock_repo_consultas
):
    """Verifica que el retorno de obtener_historial_evaluaciones incluya
    el campo resumen_usuario con el promedio calculado del usuario."""
    with patch("tools.evaluacion_ragas.RepositorioEvaluaciones", return_value=mock_repo_evaluaciones), \
         patch("tools.evaluacion_ragas.RepositorioConsultas", return_value=mock_repo_consultas):

        from tools.evaluacion_ragas import ToolsEvaluacionRAGAS
        tools = ToolsEvaluacionRAGAS()
        resultado = await tools.obtener_historial_evaluaciones(id_usuario=1)

    assert "resumen_usuario" in resultado
    resumen = resultado["resumen_usuario"]
    assert resumen is not None
    assert "promedio_general" in resumen
    assert isinstance(resumen["promedio_general"], float)


@pytest.mark.asyncio
async def test_evaluar_consulta_inexistente(
    mock_repo_evaluaciones
):
    """Verifica que evaluar_respuesta_rag retorne un error descriptivo en español
    cuando id_consulta no existe en la base de datos."""
    from database.repositorio_consultas import RepositorioConsultas
    repo_consultas = MagicMock(spec=RepositorioConsultas)
    repo_consultas.obtener_consulta_por_id = AsyncMock(return_value=None)

    with patch("tools.evaluacion_ragas.RepositorioEvaluaciones", return_value=mock_repo_evaluaciones), \
         patch("tools.evaluacion_ragas.RepositorioConsultas", return_value=repo_consultas):

        from tools.evaluacion_ragas import ToolsEvaluacionRAGAS
        tools = ToolsEvaluacionRAGAS()
        resultado = await tools.evaluar_respuesta_rag(
            id_consulta=9999,
            respuesta_generada="Alguna respuesta",
            contexto_recuperado=["Algún contexto"]
        )

    assert "error" in resultado
    assert len(resultado["error"]) > 5