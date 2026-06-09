"""
Tool MCP — Terminal de consultas híbridas guiada para el Profesor.

Se activa con /usarterminal. Recibe un bloque de texto que puede contener:
  - Líneas de comentario SQL (-- texto del intent)
  - Código SQL arbitrario DESPUÉS del comentario (completamente ignorado)

La tool lee SOLO el comentario, extrae el intent y los parámetros,
ejecuta la consulta híbrida real y devuelve el resultado con todos
los pasos del pipeline visibles.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from mcp.types import Tool

from database.embeddings.codificador_texto import CodificadorTexto
from database.repositorio_documentos import RepositorioDocumentos


# ─────────────────────────────────────────────────────────────────────────────
# Parser de comentarios SQL
# ─────────────────────────────────────────────────────────────────────────────

# Palabras que indican el valor del filtro va inline: idioma=español
_RE_FILTRO_INLINE = re.compile(
    r"\b(?P<clave>idioma|fuente|id_objeto)\s*[=:]\s*(?P<valor>\S+)",
    re.IGNORECASE,
)

# Detección de top_k: "top 5", "los 3 primeros", "5 resultados", "top_k=8"
_RE_TOP_K = re.compile(
    r"(?:top[_\s]?k?\s*[=:]?\s*|los\s+|dame\s+|)(\d+)\s*(?:resultado|primer|chunk|doc)?",
    re.IGNORECASE,
)

# Palabras que mapean a filtros de idioma
_IDIOMA_MAP = {
    "español": "es",
    "espanol": "es",
    "castellano": "es",
    "inglés": "en",
    "ingles": "en",
    "english": "en",
    "frances": "fr",
    "francés": "fr",
    "french": "fr",
    "aleman": "de",
    "alemán": "de",
    "german": "de",
    "portugues": "pt",
    "português": "pt",
}


def _extraer_comentarios(texto: str) -> str:
    """Extrae y concatena todas las líneas que empiezan con -- o /*...*/."""
    lineas = []
    for linea in texto.splitlines():
        linea_strip = linea.strip()
        if linea_strip.startswith("--"):
            # Elimina el -- y espacios iniciales
            lineas.append(linea_strip.lstrip("-").strip())
    return " ".join(lineas).strip()


def _parsear_intent(comentario: str) -> Tuple[str, Dict[str, Any], int]:
    """
    Analiza el comentario y extrae:
      - texto_busqueda: la pregunta/tema limpio para vectorizar
      - filtros: dict con pares columna→valor
      - top_k: número de resultados pedidos

    Returns:
        (texto_busqueda, filtros, top_k)
    """
    filtros: Dict[str, str] = {}
    texto_trabajo = comentario

    # ── Filtros inline: idioma=español, fuente=nasa ───────────────────────
    for m in _RE_FILTRO_INLINE.finditer(comentario):
        clave = m.group("clave").lower()
        valor = m.group("valor").strip("\"',;")
        if clave == "idioma":
            valor = _IDIOMA_MAP.get(valor.lower(), valor.lower())
        filtros[clave] = valor
        texto_trabajo = texto_trabajo.replace(m.group(0), " ")

    # ── Filtro de idioma por palabra suelta: "en español", "in english" ──
    if "idioma" not in filtros:
        for palabra, codigo in _IDIOMA_MAP.items():
            patron = re.compile(
                rf"\b(?:en\s+|in\s+)?{re.escape(palabra)}\b", re.IGNORECASE
            )
            if patron.search(texto_trabajo):
                filtros["idioma"] = codigo
                texto_trabajo = patron.sub(" ", texto_trabajo)
                break

    # ── top_k ─────────────────────────────────────────────────────────────
    top_k = 5
    m_top = _RE_TOP_K.search(texto_trabajo)
    if m_top:
        top_k = max(1, min(int(m_top.group(1)), 20))
        texto_trabajo = texto_trabajo[: m_top.start()] + texto_trabajo[m_top.end() :]

    # ── Limpiar palabras de relleno del intent ────────────────────────────
    ruido = [
        r"\bhaz\s+(esta|una|la)\s+consulta\b",
        r"\bconsulta\s+de\b",
        r"\bbusca\b",
        r"\bmuestra\b",
        r"\bcon\s+(?:filtro|parametro|parámetro)s?\b",
        r"\bcon\s+tales?\s+parametros?\b",
        r"\bde\s+la\s+forma\b",
        r"\bpara\b",
        r"\bsobre\b",
        r"\bque\b",
    ]
    for patron in ruido:
        texto_trabajo = re.sub(patron, " ", texto_trabajo, flags=re.IGNORECASE)

    texto_busqueda = re.sub(r"\s{2,}", " ", texto_trabajo).strip(" .,;:-")

    # Si el texto quedó vacío, usar el comentario original sin filtros extraídos
    if not texto_busqueda:
        texto_busqueda = comentario

    return texto_busqueda, filtros, top_k


# ─────────────────────────────────────────────────────────────────────────────
# Tool MCP
# ─────────────────────────────────────────────────────────────────────────────

class ToolsTerminalProfesor:
    """
    Terminal de consultas híbridas guiada.

    El usuario escribe un comentario SQL con el intent y (opcionalmente)
    código SQL después. La tool ignora el código, lee el comentario,
    detecta parámetros automáticamente y ejecuta la consulta híbrida real.
    """

    def __init__(self, codificador: CodificadorTexto) -> None:
        self._codificador   = codificador
        self._repo           = RepositorioDocumentos()

    def obtener_definiciones_tools(self) -> List[Tool]:
        return [
            Tool(
                name="usarterminal",
                description=(
                    "Terminal de consultas híbridas SQL+vector para demostración académica. "
                    "Úsala cuando el usuario escriba '/usarterminal' seguido de un bloque "
                    "de texto que contenga comentarios SQL con '--'. "
                    "La tool lee SOLO el comentario (ignora el código SQL), detecta el "
                    "intent y los filtros automáticamente, ejecuta la consulta híbrida "
                    "real y muestra el pipeline completo paso a paso."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entrada": {
                            "type": "string",
                            "description": (
                                "Bloque completo que escribió el usuario. Puede incluir "
                                "comentarios -- con el intent y código SQL debajo. "
                                "Solo los comentarios son procesados."
                            ),
                        }
                    },
                    "required": ["entrada"],
                },
            )
        ]

    async def usarterminal(self, entrada: str) -> Dict[str, Any]:
        t_inicio = time.monotonic()

        # ── Paso 1: Extraer comentario ────────────────────────────────────
        comentario = _extraer_comentarios(entrada)
        if not comentario:
            return {
                "error": "No se encontró ningún comentario SQL (líneas con --).",
                "sugerencia": (
                    "Escribe tu intent como comentario SQL, por ejemplo:\n"
                    "-- planetas similares a la Tierra con idioma=es\n"
                    "SELECT * FROM ...  (este código es ignorado)"
                ),
            }

        # ── Paso 2: Parsear intent ────────────────────────────────────────
        texto_busqueda, filtros, top_k = _parsear_intent(comentario)

        # ── Paso 3: Vectorizar ────────────────────────────────────────────
        t_vec = time.monotonic()
        vector = await self._codificador.codificar_texto(texto_busqueda)
        ms_vec = round((time.monotonic() - t_vec) * 1000)

        # ── Paso 4: Consulta híbrida ──────────────────────────────────────
        t_query = time.monotonic()
        resultados = await self._repo.consulta_hibrida_sql(
            vector_consulta=vector,
            filtros=filtros or {},
            top_k=top_k,
        )
        ms_query = round((time.monotonic() - t_query) * 1000)

        ms_total = round((time.monotonic() - t_inicio) * 1000)

        # ── Construir SQL visible (la misma lógica del query builder) ──────
        sql_mostrar = _sql_visible(filtros, top_k)

        # ── Respuesta enriquecida ─────────────────────────────────────────
        return {
            "presentacion_markdown": _construir_markdown(
                entrada=entrada,
                comentario=comentario,
                texto_busqueda=texto_busqueda,
                filtros=filtros,
                top_k=top_k,
                vector_dims=len(vector),
                sql_mostrar=sql_mostrar,
                resultados=resultados,
                ms_vec=ms_vec,
                ms_query=ms_query,
                ms_total=ms_total,
            ),
            "mensaje_para_claude": (
                "Renderiza el campo presentacion_markdown completo con todo el formato Markdown. "
                "Muestra el pipeline paso a paso exactamente como está, con las tablas y "
                "bloques de código. Al final comenta brevemente qué tan relevantes parecen "
                "los resultados según el intent original del comentario. "
                "RECUERDA: antes de cada respuesta muestra el bloque ⚙️ Pipeline ejecutado."
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de presentación
# ─────────────────────────────────────────────────────────────────────────────

def _sql_visible(filtros: Dict[str, Any], top_k: int) -> str:
    """Genera el SQL parametrizado que se habría ejecutado."""
    condiciones = []
    param_idx = 3
    param_ejemplos = {}
    for clave, valor in filtros.items():
        condiciones.append(f"d.{clave} = ${param_idx}   -- '{valor}'")
        param_ejemplos[f"${param_idx}"] = repr(valor)
        param_idx += 1

    where = ""
    if condiciones:
        where = "WHERE\n    " + "\n    AND ".join(condiciones) + "\n"

    return f"""SELECT
    et.id_doc,
    d.titulo,
    et.chunk_id,
    et.contenido_chunk              AS contenido,
    et.estrategia_chunking,
    1 - (et.vector <=> $1::vector)  AS similitud
FROM Embedding_Texto et
JOIN Documento d ON d.id_doc = et.id_doc
{where}ORDER BY et.vector <=> $1::vector   -- orden por distancia coseno
LIMIT $2                                   -- $2 = {top_k}"""


def _construir_markdown(
    entrada: str,
    comentario: str,
    texto_busqueda: str,
    filtros: Dict[str, Any],
    top_k: int,
    vector_dims: int,
    sql_mostrar: str,
    resultados: List[dict],
    ms_vec: int,
    ms_query: int,
    ms_total: int,
) -> str:
    # ── Sección de filtros detectados ─────────────────────────────────────
    if filtros:
        filas_filtros = "\n".join(
            f"| `{k}` | `{v}` | WHERE d.{k} = ... |"
            for k, v in filtros.items()
        )
        tabla_filtros = (
            f"| Parámetro | Valor detectado | Cláusula SQL |\n"
            f"|:---:|:---:|:---|\n"
            f"{filas_filtros}\n"
        )
    else:
        tabla_filtros = "*Sin filtros relacionales — búsqueda semántica pura.*\n"

    # ── Sección de resultados ─────────────────────────────────────────────
    if resultados:
        filas_res = "\n".join(
            f"| {i+1} | **{r['titulo']}** "
            f"| chunk {r['chunk_id']} "
            f"| {r['similitud']:.4f} "
            f"| {r.get('estrategia_chunking') or 'N/D'} |"
            for i, r in enumerate(resultados)
        )
        tabla_res = (
            f"| # | Documento | Chunk | Similitud | Estrategia |\n"
            f"|:---:|:---|:---:|:---:|:---:|\n"
            f"{filas_res}\n"
        )

        # Detalle del primer resultado
        primer = resultados[0]
        contenido_preview = (primer.get("contenido") or "")[:300]
        if len(primer.get("contenido") or "") > 300:
            contenido_preview += "…"

        detalle_primer = (
            f"### 📄 Chunk más relevante\n\n"
            f"> **{primer['titulo']}** — chunk {primer['chunk_id']} "
            f"(similitud: `{primer['similitud']:.4f}`)\n\n"
            f"> {contenido_preview}\n\n"
        ) if contenido_preview else ""
    else:
        tabla_res = "*No se encontraron resultados para este intent.*\n"
        detalle_primer = ""

    # ── Entrada original (solo el comentario, no el código) ───────────────
    lineas_comentario = [
        l.strip() for l in entrada.splitlines() if l.strip().startswith("--")
    ]
    bloque_entrada = "\n".join(lineas_comentario)

    return (
        f"# 🖥️ Terminal de Consultas Híbridas\n\n"
        f"---\n\n"
        f"## 📝 Paso 1 — Lectura del comentario\n\n"
        f"```sql\n{bloque_entrada}\n```\n\n"
        f"> ℹ️ El código SQL escrito después del comentario fue **ignorado**.\n"
        f"> Solo el intent del comentario es procesado por la tool.\n\n"
        f"---\n\n"
        f"## 🧠 Paso 2 — Intent extraído\n\n"
        f"| Campo | Valor |\n"
        f"|:---|:---|\n"
        f"| 🔍 Texto a vectorizar | **`{texto_busqueda}`** |\n"
        f"| 🔢 Top K | **{top_k}** resultados |\n\n"
        f"**Filtros relacionales detectados:**\n\n"
        f"{tabla_filtros}\n"
        f"---\n\n"
        f"## 🔢 Paso 3 — Vectorización\n\n"
        f"| Modelo | Dimensiones | Tiempo |\n"
        f"|:---|:---:|:---:|\n"
        f"| `all-MiniLM-L6-v2` | **{vector_dims}** | **{ms_vec} ms** |\n\n"
        f"> El texto `\"{texto_busqueda}\"` fue convertido a un vector de **{vector_dims} dimensiones**.\n"
        f"> Este vector representa la posición semántica de la pregunta en el espacio vectorial.\n\n"
        f"---\n\n"
        f"## 🗄️ Paso 4 — Query SQL construida por el query builder\n\n"
        f"```sql\n{sql_mostrar}\n```\n\n"
        f"> **¿Por qué una sola query?** El `WHERE` reduce el espacio de búsqueda relacionalmente\n"
        f"> antes de que pgvector calcule las distancias coseno (`<=>`). Es más eficiente\n"
        f"> que filtrar en Python después de recuperar todos los embeddings.\n\n"
        f"---\n\n"
        f"## 📦 Paso 5 — Resultados ({len(resultados)} de {top_k} solicitados)\n\n"
        f"{tabla_res}\n"
        f"{detalle_primer}"
        f"---\n\n"
        f"## ⏱️ Resumen de tiempos\n\n"
        f"| Fase | Tiempo |\n"
        f"|:---|:---:|\n"
        f"| Vectorización (MiniLM) | **{ms_vec} ms** |\n"
        f"| Query PostgreSQL + pgvector | **{ms_query} ms** |\n"
        f"| **Total end-to-end** | **{ms_total} ms** |\n\n"
        f"---\n"
        f"*AstroData Lab · Nicolás · Valeria · Jeferson · Bases de Datos Relacionales*"
    )
