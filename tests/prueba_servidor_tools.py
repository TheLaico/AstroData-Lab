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
    assert "astro_data_lab" in nombres
    assert "rag_query" in nombres
    assert "crear_objeto_astronomico" in nombres
    assert "generar_embeddings_imagenes_pendientes" in nombres
    assert "reemplazar_imagen_con_embedding" in nombres
    assert "eliminar_imagen_astronomica" in nombres
    assert "encontrar_planetas_similares" in nombres
    assert "evaluar_respuesta_rag" in nombres


@pytest.mark.asyncio
async def test_servidor_mcp_registra_y_enruta_tools_de_imagen():
    from server.servidor_mcp import (
        GrupoBusquedaSemantica,
        GrupoGestionObjetos,
        GrupoPresentacion,
        ServidorMCPAstroData,
    )

    codificador_texto = MagicMock()
    codificador_texto.codificar_texto = AsyncMock(return_value=[0.1] * 384)
    codificador_texto.nombre_modelo = AsyncMock(return_value="all-MiniLM-L6-v2")

    codificador_imagen = MagicMock()
    codificador_imagen.codificar_texto = AsyncMock(return_value=[0.2] * 512)
    codificador_imagen.codificar_imagen = AsyncMock(return_value=[0.3] * 512)
    codificador_imagen.nombre_modelo = AsyncMock(return_value="openai/clip-vit-base-patch32")

    servidor = ServidorMCPAstroData()
    await servidor._inicializar_herramientas(codificador_texto, codificador_imagen)
    servidor._registrar_tools()

    tools_gestion = {
        "crear_imagen_con_embedding",
        "generar_embeddings_imagenes_pendientes",
        "reemplazar_imagen_con_embedding",
        "eliminar_imagen_astronomica",
    }
    tools_busqueda = {
        "buscar_imagenes_por_descripcion",
        "buscar_imagenes_similares",
        "obtener_info_objeto_por_imagen",
    }

    assert tools_gestion | tools_busqueda <= set(servidor._mapa_tools)
    assert isinstance(servidor._mapa_tools["astro_data_lab"], GrupoPresentacion)
    assert all(isinstance(servidor._mapa_tools[nombre], GrupoGestionObjetos) for nombre in tools_gestion)
    assert all(isinstance(servidor._mapa_tools[nombre], GrupoBusquedaSemantica) for nombre in tools_busqueda)


@pytest.mark.asyncio
async def test_tool_astro_data_lab_entrega_presentacion():
    from tools.presentacion import ToolsPresentacionAstroData

    tools = ToolsPresentacionAstroData()
    resultado = await tools.astro_data_lab(modo="corto")

    assert resultado["titulo"] == "ASTRODATA LAB"
    assert "presentacion_markdown" in resultado
    assert "preguntas_demo" in resultado
    assert "Claude" in resultado["presentacion_markdown"]
