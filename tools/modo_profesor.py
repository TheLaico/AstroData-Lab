"""Tool MCP — Modo demostración guiada para el Profesor Paolo."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from mcp.types import Tool
from database.conexion import conexion_bd
from config.ajustes import ajustes


class ToolsModoProfesor:

    def obtener_definiciones_tools(self) -> List[Tool]:
        return [
            Tool(
                name="modo_profesor",
                description=(
                    "Activa la presentación formal de AstroData Lab ante el Profesor Paolo. "
                    "Úsala cuando el usuario escriba 'modo profesor', 'presentar al profesor', "
                    "'professor mode', 'demo profesor', '/modotexterprofesor' o cualquier "
                    "variación que indique presentar el proyecto a un evaluador académico."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "nombre_profesor": {
                            "type": "string",
                            "default": "Paolo",
                            "description": "Nombre del profesor.",
                        }
                    },
                },
            )
        ]

    async def modo_profesor(self, nombre_profesor: str = "Paolo") -> Dict[str, Any]:
        nombre = (nombre_profesor or "Paolo").strip()
        estado = await self._obtener_estado()

        return {
            "titulo": "ASTRODATA LAB — Modo Profesor",
            "presentacion_markdown": self._construir_presentacion(estado, nombre),
            "estado_sistema": estado,
            "consulta_inicial_sugerida": (
                "¿Qué planetas tienen condiciones similares a la Tierra?"
            ),
            "preguntas_demo": [
                "¿Qué planetas tienen condiciones similares a la Tierra?",
                "Busca documentos sobre habitabilidad planetaria en español",
                "Lista los planetas con habitabilidad mayor a 0.7",
                "¿Qué lunas del sistema solar podrían tener océanos internos?",
                "Evalúa qué tan buena fue la respuesta anterior con métricas RAGAS",
                "Busca imágenes astronómicas similares a: nebulosa con colores brillantes",
            ],
            "mensaje_para_claude": (
                f"Presenta este resultado como apertura formal ante el Profesor {nombre}. "
                "Renderiza el campo presentacion_markdown completo con todo el formato Markdown. "
                "Muestra tablas, separadores y bloques de código exactamente como están. "
                "Después de mostrar la presentación completa, SIN esperar input del profesor, "
                "ejecuta inmediatamente la tool rag_query con el texto de consulta_inicial_sugerida "
                "y muestra el resultado como primera demostración en vivo. "
                "Antes de mostrar el resultado RAG, incluye el bloque ⚙️ Pipeline ejecutado. "
                "COMPORTAMIENTO PERMANENTE para el resto de la sesión: antes de CADA respuesta "
                "incluye el bloque '---\\n⚙️ **Pipeline ejecutado**' con: "
                "🧹 Consulta procesada | "
                "🔢 Vectorización (modelo + dims) | "
                "🔧 Tool MCP invocada | "
                "🗄️ Query en PostgreSQL | "
                "📦 Construcción de respuesta. "
                "Luego responde con formato profesional."
            ),
        }

    # ── Estado completo con datos reales ─────────────────────────────────────

    async def _obtener_estado(self) -> Dict[str, Any]:
        estado: Dict[str, Any] = {
            "base_datos": "no verificada",
            "modelo_texto":  ajustes.modelo_texto,
            "modelo_imagen": ajustes.modelo_imagen,
            "metricas": {},
            "objetos_muestra": [],
            "planetas_muestra": [],
            "documentos_muestra": [],
        }

        consultas_conteo = {
            "objetos_astronomicos":  "SELECT COUNT(*) FROM Objeto_Astronomico",
            "documentos":            "SELECT COUNT(*) FROM Documento",
            "imagenes":              "SELECT COUNT(*) FROM Imagen",
            "embeddings_texto":      "SELECT COUNT(*) FROM Embedding_Texto",
            "embeddings_imagen":     "SELECT COUNT(*) FROM Embedding_Imagen",
            "consultas_rag":         "SELECT COUNT(*) FROM Consulta",
            "evaluaciones_ragas":    "SELECT COUNT(*) FROM Evaluacion_RAGAS",
        }

        try:
            async with conexion_bd.obtener_conexion() as conn:

                # Conteos generales
                for clave, sql in consultas_conteo.items():
                    try:
                        estado["metricas"][clave] = await conn.fetchval(sql)
                    except Exception:
                        estado["metricas"][clave] = "N/D"

                # Imágenes sin embedding
                try:
                    estado["metricas"]["imagenes_sin_embedding"] = await conn.fetchval(
                        """
                        SELECT COUNT(*) FROM Imagen i
                        LEFT JOIN Embedding_Imagen ei ON ei.id_imagen = i.id_imagen
                        WHERE ei.id_embedding IS NULL
                        """
                    )
                except Exception:
                    estado["metricas"]["imagenes_sin_embedding"] = "N/D"

                # Muestra de objetos astronómicos reales
                try:
                    filas = await conn.fetch(
                        """
                        SELECT nombre,
                               LEFT(descripcion_cientifica, 80) AS desc_corta
                        FROM Objeto_Astronomico
                        WHERE descripcion_cientifica IS NOT NULL
                        ORDER BY id_objeto
                        LIMIT 6
                        """
                    )
                    estado["objetos_muestra"] = [
                        {"nombre": f["nombre"], "descripcion": f["desc_corta"]}
                        for f in filas
                    ]
                except Exception:
                    estado["objetos_muestra"] = []

                # Muestra de planetas con habitabilidad
                try:
                    filas = await conn.fetch(
                        """
                        SELECT o.nombre,
                               p.puntaje_habitabilidad,
                               p.tiene_agua,
                               p.tipo_atmosfera
                        FROM Planeta p
                        JOIN Objeto_Astronomico o ON o.id_objeto = p.id_objeto
                        ORDER BY p.puntaje_habitabilidad DESC NULLS LAST
                        LIMIT 5
                        """
                    )
                    estado["planetas_muestra"] = [
                        {
                            "nombre":        f["nombre"],
                            "habitabilidad": float(f["puntaje_habitabilidad"]) if f["puntaje_habitabilidad"] else None,
                            "tiene_agua":    f["tiene_agua"],
                            "atmosfera":     f["tipo_atmosfera"],
                        }
                        for f in filas
                    ]
                except Exception:
                    estado["planetas_muestra"] = []

                # Muestra de documentos recientes
                try:
                    filas = await conn.fetch(
                        """
                        SELECT d.titulo,
                               d.idioma,
                               d.fuente,
                               o.nombre AS objeto_relacionado
                        FROM Documento d
                        LEFT JOIN Objeto_Astronomico o ON o.id_objeto = d.id_objeto
                        ORDER BY d.id_doc DESC
                        LIMIT 4
                        """
                    )
                    estado["documentos_muestra"] = [
                        {
                            "titulo":   f["titulo"],
                            "idioma":   f["idioma"],
                            "fuente":   f["fuente"],
                            "objeto":   f["objeto_relacionado"],
                        }
                        for f in filas
                    ]
                except Exception:
                    estado["documentos_muestra"] = []

                estado["base_datos"] = "conectada"

        except Exception as exc:
            estado["base_datos"] = "sin conexion"
            estado["error"] = str(exc)

        return estado

    # ── Presentación visual ───────────────────────────────────────────────────

    def _construir_presentacion(self, estado: Dict[str, Any], nombre: str) -> str:
        m          = estado.get("metricas", {})
        objetos    = m.get("objetos_astronomicos", "N/D")
        docs       = m.get("documentos", "N/D")
        emb_texto  = m.get("embeddings_texto", "N/D")
        imagenes   = m.get("imagenes", "N/D")
        emb_img    = m.get("embeddings_imagen", "N/D")
        consultas  = m.get("consultas_rag", "N/D")
        evals      = m.get("evaluaciones_ragas", "N/D")
        bd         = estado.get("base_datos", "N/D")
        mod_txt    = estado.get("modelo_texto", "N/D")
        mod_img    = estado.get("modelo_imagen", "N/D")

        # Saludo según hora del día
        hora = datetime.now().hour
        if hora < 12:
            saludo = "Buenos días"
        elif hora < 18:
            saludo = "Buenas tardes"
        else:
            saludo = "Buenas noches"

        # Sección de objetos reales
        objetos_muestra   = estado.get("objetos_muestra", [])
        planetas_muestra  = estado.get("planetas_muestra", [])
        documentos_muestra = estado.get("documentos_muestra", [])

        seccion_objetos = ""
        if objetos_muestra:
            filas = "\n".join(
                f"| 🌠 **{o['nombre']}** | {o['descripcion']}… |"
                for o in objetos_muestra
            )
            seccion_objetos = (
                f"## 🔭 Objetos astronómicos en la base de datos\n\n"
                f"| Nombre | Descripción científica |\n"
                f"|:---|:---|\n"
                f"{filas}\n\n"
                f"---\n\n"
            )

        seccion_planetas = ""
        if planetas_muestra:
            filas = "\n".join(
                f"| 🪐 **{p['nombre']}** "
                f"| {f\"{p['habitabilidad']:.2f}\" if p['habitabilidad'] is not None else 'N/D'} "
                f"| {'✅' if p['tiene_agua'] else '❌'} "
                f"| {p['atmosfera'] or 'N/D'} |"
                for p in planetas_muestra
            )
            seccion_planetas = (
                f"## 🌍 Planetas — ranking de habitabilidad\n\n"
                f"| Planeta | Puntaje | Agua | Atmósfera |\n"
                f"|:---|:---:|:---:|:---|\n"
                f"{filas}\n\n"
                f"---\n\n"
            )

        seccion_docs = ""
        if documentos_muestra:
            filas = "\n".join(
                f"| 📄 **{d['titulo']}** | {d['idioma'] or 'N/D'} | {d['objeto'] or '—'} |"
                for d in documentos_muestra
            )
            seccion_docs = (
                f"## 📚 Documentos científicos recientes\n\n"
                f"| Título | Idioma | Objeto relacionado |\n"
                f"|:---|:---:|:---|\n"
                f"{filas}\n\n"
                f"---\n\n"
            )

        return (
            f"# 🌌 AstroData Lab\n"
            f"### Sistema Híbrido para Exploración Inteligente del Universo\n\n"
            f"---\n\n"
            f"> 👋 **{saludo}, Profesor {nombre}.**\n"
            f">\n"
            f"> Bienvenido a la demostración en vivo de **AstroData Lab** — un sistema que conecta\n"
            f"> **Claude Desktop** con una base de datos **PostgreSQL real** enriquecida con\n"
            f"> búsqueda vectorial mediante **pgvector**. Todo lo que verá aquí consulta datos reales.\n\n"
            f"---\n\n"
            f"## 📡 Estado del sistema — ahora mismo\n\n"
            f"| Componente | Estado |\n"
            f"|:---|:---:|\n"
            f"| 🗄️ Base de datos PostgreSQL | **{bd}** |\n"
            f"| 🪐 Objetos astronómicos | **{objetos}** |\n"
            f"| 📄 Documentos científicos | **{docs}** |\n"
            f"| 🔢 Chunks vectorizados | **{emb_texto}** |\n"
            f"| 🖼️ Imágenes astronómicas | **{imagenes}** |\n"
            f"| 👁️ Embeddings visuales CLIP | **{emb_img}** |\n"
            f"| 💬 Consultas RAG registradas | **{consultas}** |\n"
            f"| 📈 Evaluaciones RAGAS | **{evals}** |\n\n"
            f"---\n\n"
            f"## 🧠 Modelos de embeddings activos\n\n"
            f"| Modalidad | Modelo | Dims | Uso |\n"
            f"|:---:|:---|:---:|:---|\n"
            f"| 📝 Texto | `{mod_txt}` | **384** | RAG + búsqueda semántica |\n"
            f"| 🖼️ Imagen | `{mod_img}` | **512** | Búsqueda visual multimodal |\n\n"
            f"---\n\n"
            f"## 🏗️ Arquitectura del sistema\n\n"
            f"**Capas:**  `Claude Desktop` → `MCP` → `Tools` → `Services` → `Repositorios` → `PostgreSQL + pgvector`\n\n"
            f"**Pipeline RAG:**  `Pregunta` → `MiniLM 384d` → `coseno <=>` → `chunks` → `respuesta`\n\n"
            f"**Consulta híbrida:**  `WHERE relacional` + `ORDER BY vector <=>` en **una sola query SQL**\n\n"
            f"---\n\n"
            f"{seccion_objetos}"
            f"{seccion_planetas}"
            f"{seccion_docs}"
            f"## 🎯 Capacidades disponibles\n\n"
            f"| # | Capacidad | Tecnología |\n"
            f"|:---:|:---|:---|\n"
            f"| 1️⃣ | **Consulta RAG** — pregunta científica libre | MiniLM + pgvector |\n"
            f"| 2️⃣ | **Consulta Híbrida** — SQL + vector en una query | Query builder + `<=>` |\n"
            f"| 3️⃣ | **Planetas habitables** — capa relacional pura | SELECT directo |\n"
            f"| 4️⃣ | **Búsqueda de imágenes** — por texto o URL | CLIP 512d |\n"
            f"| 5️⃣ | **Evaluación RAGAS** — calidad de respuesta | faithfulness · relevancy · recall |\n"
            f"| 6️⃣ | **Contexto de objeto** — todo sobre un cuerpo celeste | JOIN completo |\n\n"
            f"---\n\n"
            f"## ✨ Preguntas recomendadas\n\n"
            f"➤ *\"¿Qué planetas tienen condiciones similares a la Tierra?\"*\n"
            f"➤ *\"Busca documentos sobre habitabilidad planetaria en español\"*\n"
            f"➤ *\"Lista los planetas con habitabilidad mayor a 0.7\"*\n"
            f"➤ *\"¿Qué lunas podrían tener océanos internos?\"*\n"
            f"➤ *\"Evalúa la calidad de la respuesta anterior\"*\n\n"
            f"---\n\n"
            f"> 💡 **Primera consulta RAG ejecutándose ahora mismo** ↓\n\n"
            f"---\n"
            f"*AstroData Lab · Nicolás · Valeria · Jeferson · Bases de Datos Relacionales*"
        )
