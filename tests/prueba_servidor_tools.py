import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_servidor_registra_tools_mvp():
    from server.servidor_mcp import ServidorMCPAstroData

    codificador = MagicMock()
    codificador.codificar_texto = AsyncMock(return_value=[0.1] * 384)
    codificador.nombre_modelo = AsyncMock(return_value="all-MiniLM-L6-v2")

    servidor = ServidorMCPAstroData()
    await servidor._inicializar_herramientas(codificador, codificador)
    servidor._registrar_tools()

    nombres = {tool.name for tool in servidor._tool_defs}
    assert "rag_query" in nombres
    assert "crear_objeto_astronomico" in nombres
    assert "encontrar_planetas_similares" in nombres
    assert "evaluar_respuesta_rag" in nombres
