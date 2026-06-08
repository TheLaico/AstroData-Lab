"""Tool MCP de bienvenida y demo para AstroData Lab."""

from typing import Any, Dict, List

from mcp.types import Tool

from config.ajustes import ajustes
from database.conexion import conexion_bd


class ToolsPresentacionAstroData:
    """Presentacion rapida del proyecto para demostraciones en Claude Desktop."""

    def obtener_definiciones_tools(self) -> List[Tool]:
        return [
            Tool(
                name="astro_data_lab",
                description=(
                    "Muestra una bienvenida visual e impactante de AstroData Lab. "
                    "Usala cuando el usuario escriba 'astro data lab', quiera iniciar "
                    "la demo, presentar el proyecto al profesor o ver un resumen del sistema."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "modo": {
                            "type": "string",
                            "default": "profesor",
                            "description": "Estilo de presentacion: profesor, tecnico o corto.",
                        }
                    },
                },
            )
        ]

    async def astro_data_lab(self, modo: str = "profesor") -> Dict[str, Any]:
        modo_normalizado = (modo or "profesor").strip().lower()
        estado = await self._obtener_estado()

        return {
            "titulo": "ASTRODATA LAB",
            "subtitulo": "Laboratorio inteligente de datos astronomicos con RAG, pgvector y MCP",
            "modo": modo_normalizado,
            "presentacion_markdown": self._construir_presentacion(estado, modo_normalizado),
            "estado_sistema": estado,
            "preguntas_demo": [
                "Busca imagenes astronomicas similares a: planeta gigante con anillos brillantes",
                "Busca imagenes similares a esta URL: https://upload.wikimedia.org/wikipedia/commons/c/c7/Saturn_during_Equinox.jpg",
                "Haz una consulta RAG sobre TRAPPIST-1 y explica el contexto recuperado",
                "Lista planetas con puntaje de habitabilidad minimo de 0.7",
                "Evalua una respuesta RAG y dime su calidad general",
            ],
            "mensaje_para_claude": (
                "Presenta este resultado como apertura de demo. Usa formato elegante, "
                "breve y convincente, resaltando que el proyecto conecta Claude Desktop "
                "con PostgreSQL, pgvector, embeddings de texto e imagen y herramientas MCP reales."
            ),
        }

    async def _obtener_estado(self) -> Dict[str, Any]:
        estado: Dict[str, Any] = {
            "base_datos": "no verificada",
            "modelo_texto": ajustes.modelo_texto,
            "modelo_imagen": ajustes.modelo_imagen,
            "top_k": ajustes.top_k,
            "metricas": {},
        }

        consultas = {
            "objetos_astronomicos": "SELECT COUNT(*) FROM Objeto_Astronomico",
            "documentos": "SELECT COUNT(*) FROM Documento",
            "imagenes": "SELECT COUNT(*) FROM Imagen",
            "embeddings_texto": "SELECT COUNT(*) FROM Embedding_Texto",
            "embeddings_imagen": "SELECT COUNT(*) FROM Embedding_Imagen",
            "consultas_rag": "SELECT COUNT(*) FROM Consulta",
            "evaluaciones_ragas": "SELECT COUNT(*) FROM Evaluacion_RAGAS",
        }

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                for nombre, sql in consultas.items():
                    try:
                        estado["metricas"][nombre] = await conexion.fetchval(sql)
                    except Exception:
                        estado["metricas"][nombre] = "no disponible"

                pendientes = await conexion.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM Imagen i
                    LEFT JOIN Embedding_Imagen ei ON ei.id_imagen = i.id_imagen
                    WHERE ei.id_embedding IS NULL
                    """
                )
                estado["metricas"]["imagenes_sin_embedding"] = pendientes
                estado["base_datos"] = "conectada"
        except Exception as exc:
            estado["base_datos"] = "sin conexion"
            estado["error"] = str(exc)

        return estado

    def _construir_presentacion(self, estado: Dict[str, Any], modo: str) -> str:
        metricas = estado.get("metricas", {})
        imagenes = metricas.get("imagenes", "N/D")
        embeddings_imagen = metricas.get("embeddings_imagen", "N/D")
        docs = metricas.get("documentos", "N/D")
        embeddings_texto = metricas.get("embeddings_texto", "N/D")
        objetos = metricas.get("objetos_astronomicos", "N/D")
        pendientes = metricas.get("imagenes_sin_embedding", "N/D")

        if modo == "corto":
            return (
                "# ASTRODATA LAB\n\n"
                "Claude conectado a un laboratorio astronomico con PostgreSQL, "
                "pgvector, RAG, embeddings CLIP de imagen y herramientas MCP reales.\n\n"
                f"Estado: BD {estado.get('base_datos')} | Objetos: {objetos} | "
                f"Docs: {docs} | Imagenes: {imagenes} | Pendientes: {pendientes}"
            )

        detalle_tecnico = (
            "- MCP sobre stdio expone funciones invocables desde Claude Desktop.\n"
            "- PostgreSQL guarda objetos, documentos, consultas, resultados e imagenes.\n"
            "- pgvector permite busqueda semantica por texto y similitud visual.\n"
            f"- Texto: {estado.get('modelo_texto')}.\n"
            f"- Imagen: {estado.get('modelo_imagen')}.\n"
        )

        cierre = (
            "## Demo sugerida\n"
            "1. Buscar por texto: `planeta gigante con anillos brillantes`.\n"
            "2. Buscar por imagen usando una URL publica de Saturno.\n"
            "3. Ejecutar una consulta RAG sobre un sistema planetario.\n"
            "4. Evaluar la respuesta con metricas RAGAS simplificadas.\n"
        )

        if modo == "tecnico":
            return (
                "# ASTRODATA LAB - Vista tecnica\n\n"
                "Sistema RAG astronomico conectado a Claude Desktop mediante MCP.\n\n"
                "## Arquitectura\n"
                f"{detalle_tecnico}\n"
                "## Estado actual\n"
                f"- Base de datos: {estado.get('base_datos')}.\n"
                f"- Objetos astronomicos: {objetos}.\n"
                f"- Documentos: {docs}.\n"
                f"- Embeddings de texto: {embeddings_texto}.\n"
                f"- Imagenes: {imagenes}.\n"
                f"- Embeddings de imagen: {embeddings_imagen}.\n"
                f"- Imagenes sin embedding: {pendientes}.\n\n"
                f"{cierre}"
            )

        return (
            "# ASTRODATA LAB\n\n"
            "**Un laboratorio de datos astronomicos conectado directamente a Claude.**\n\n"
            "Este proyecto no solo responde preguntas: consulta una base relacional, "
            "recupera contexto cientifico, busca documentos por significado y compara "
            "imagenes astronomicas con embeddings CLIP.\n\n"
            "## Lo que esta vivo ahora\n"
            f"- Base de datos: **{estado.get('base_datos')}**.\n"
            f"- Objetos astronomicos registrados: **{objetos}**.\n"
            f"- Documentos cientificos: **{docs}**.\n"
            f"- Embeddings de texto: **{embeddings_texto}**.\n"
            f"- Imagenes astronomicas: **{imagenes}**.\n"
            f"- Embeddings visuales CLIP: **{embeddings_imagen}**.\n"
            f"- Imagenes pendientes de vectorizar: **{pendientes}**.\n\n"
            "## Capacidades estrella\n"
            "- Preguntas RAG con contexto recuperado desde PostgreSQL.\n"
            "- Busqueda semantica de documentos con embeddings.\n"
            "- Busqueda de imagenes por descripcion textual.\n"
            "- Busqueda de imagenes similares desde ruta, URL o base64.\n"
            "- CRUD de objetos astronomicos, telescopios y observaciones.\n"
            "- Evaluacion RAGAS simplificada para medir calidad de respuestas.\n\n"
            f"{cierre}"
        )
