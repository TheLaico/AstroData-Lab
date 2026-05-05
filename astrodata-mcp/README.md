
---

## Contexto: AstroData Lab — Sistema RAG Híbrido de Exploración Astronómica

**Autores:** Nicolas Vargas, Valeria Londoño | **Curso:** Bases de Datos Relacionales | 

---

### ¿Qué es AstroData Lab?

AstroData Lab es un sistema de base de datos híbrido (relacional + vectorial) diseñado para explorar y analizar datos astronómicos con el objetivo de estudiar **condiciones de habitabilidad en el universo**. Gestiona una jerarquía de cuerpos celestes: Galaxias → Sistemas Estelares → Estrellas → Planetas → Lunas.

El problema central que resuelve es la coexistencia de **datos estructurados** (atributos físicos exactos) y **datos semánticos** (texto científico, imágenes), integrándolos mediante un pipeline **RAG (Retrieval-Augmented Generation)**: recupera información relevante desde la BD relacional y la BD vectorial, y usa un LLM para generar respuestas contextualizadas en lenguaje natural.

---

### Arquitectura general

El sistema tiene **dos componentes principales**:

1. **Modelo Relacional (PostgreSQL)** — datos estructurados jerárquicos sobre cuerpos celestes, documentos, observaciones, usuarios y consultas.
2. **Modelo Vectorial (pgvector)** — embeddings de texto e imágenes para búsqueda por similitud semántica.

---

### Esquema de base de datos

#### Tablas relacionales (PostgreSQL)

**Jerarquía astronómica (patrón IS-A — `Objeto_Astronomico` como supertipo):**
- `Objeto_Astronomico(id_objeto PK, nombre, descripcion_cientifica)` — entidad base para todos los cuerpos celestes
- `Galaxia(id_objeto PK→OA, id_tipo_galaxia FK, distancia_años_luz)`
- `Sistema_Estelar(id_objeto PK→OA, id_galaxia FK)`
- `Estrella(id_objeto PK→OA, id_tipo_estrella FK, id_sistema FK, masa_masas_solares, temperatura_K)`
- `Planeta(id_objeto PK→OA, id_tipo_planeta FK, id_sistema FK, masa_masas_terrestres, temperatura_K)`
- `Luna(id_objeto PK→OA, id_planeta FK, radio_km)`

**Catálogos de tipos:**
- `Tipo_Galaxia(id, nombre_tipo)` — ej: Espiral, Elíptica, Irregular
- `Tipo_Estrella(id, nombre_tipo)` — ej: Enana amarilla, Gigante roja, Enana blanca
- `Tipo_Planeta(id, nombre_tipo)` — ej: Rocoso, Gaseoso, Oceánico, Desértico

**Habitabilidad:**
- `Caracteristica_Ambiental(id PK, id_planeta FK, tipo, valor)` — ej: tipo='agua_liquida', valor='presente'
- `Evaluacion_Habitabilidad(id PK, id_planeta FK, puntaje 0-1, descripcion, fecha)`

**Documentos e imágenes:**
- `Documento(id_doc PK, titulo, idioma, fecha, fuente, contenido_texto, id_objeto FK)`
- `Imagen(id_imagen PK, ruta_archivo, descripcion, etiquetas, id_doc FK)`

**Telescopios y observaciones:**
- `Telescopio(id PK, nombre, tipo, ubicacion)`
- `Observacion(id PK, id_telescopio FK, id_objeto FK, fecha, descripcion)`

**Usuarios y consultas (pipeline RAG):**
- `Usuario(id PK, nombre, correo, fecha_registro)`
- `Consulta(id PK, texto_pregunta, fecha, id_usuario FK)`
- `Resultado(id PK, descripcion_resultado, relevancia 0-1, id_consulta FK, id_doc FK, id_imagen FK)`
- `Evaluacion(id PK, faithfulness 0-1, answer_relevancy 0-1, context_recall 0-1, modelo_eval, fecha, id_consulta FK)` — métricas RAGAS

#### Tablas vectoriales (pgvector)

- `Embedding_Texto(id PK, id_doc FK, chunk_id, estrategia_chunking, vector(384), modelo)` — índice IVFFlat coseno, lists=100
- `Embedding_Imagen(id PK, id_imagen FK, vector(512), modelo)` — índice IVFFlat coseno, lists=50
- `Embedding_Consulta(id PK, id_consulta FK, vector(384), modelo)` — índice IVFFlat coseno, lists=50

**Campos vectorizables** (usados para búsqueda semántica):
- `Consulta.texto_pregunta` → input del pipeline RAG
- `Documento.contenido_texto` y `Documento.titulo`
- `Objeto_Astronomico.descripcion_cientifica` → permite búsquedas como "planeta similar a la Tierra"
- `Imagen.descripcion` e `Imagen.etiquetas`
- `Imagen.ruta_archivo` (vectorización visual con CLIP, vector 512d)
- `Observacion.descripcion`

**Consulta vectorial típica (buscar top-5 chunks más similares a una pregunta):**
```sql
SELECT d.titulo, e.chunk_id, e.estrategia_chunking,
       1 - (e.vector <=> $1::vector) AS similitud
FROM Embedding_Texto e
JOIN Documento d ON e.id_doc = d.id_doc
ORDER BY e.vector <=> $1::vector
LIMIT 5;
-- $1 = embedding del texto de la consulta del usuario
```

---

### Normalización

Todas las tablas están en **Tercera Forma Normal (3FN)**. Las decisiones clave fueron:
- Separar catálogos de tipos (Tipo_Galaxia, Tipo_Estrella, Tipo_Planeta) para evitar dependencias transitivas.
- Tablas de embeddings independientes para permitir múltiples representaciones por documento/imagen.
- `Caracteristica_Ambiental` como entidad extensible (múltiples características por planeta).
- Patrón IS-A con `Objeto_Astronomico` como supertipo para centralizar nombre y descripción.

---

### Experimento de Chunking

Se comparan dos estrategias para fragmentar textos antes de vectorizar:

| Estrategia | Tamaño | Overlap | Criterio de corte |
|---|---|---|---|
| **A — Fixed-size** | 256 tokens | 32 tokens | Por caracteres fijos (línea base) |
| **B — Sentence-based** | 128–512 tokens (variable) | 1-2 oraciones | Punto final + salto de línea |

**Hipótesis:** La estrategia B superará a la A porque las descripciones astronómicas están estructuradas en oraciones semánticamente completas.

**Métricas de evaluación (RAGAS):** `faithfulness`, `answer_relevancy`, `context_recall`.

---

### Datos de ejemplo cargados

Jerarquía de prueba: Vía Láctea → Sistema Solar → Sol → Tierra → Luna, con características ambientales de la Tierra (agua líquida, 21% oxígeno, campo magnético) y evaluación de habitabilidad = 1.0.

---
---

## SECCIÓN 6: IMPLEMENTACIÓN MCP (Model Context Protocol) EN CLAUDE DESKTOP

### 6.1 ¿Qué es el MCP en este contexto?

El sistema AstroData Lab expone sus capacidades a Claude mediante un **servidor MCP (Model Context Protocol)**, que actúa como puente entre Claude Desktop y la infraestructura de base de datos del proyecto (PostgreSQL + pgvector). Esto permite que Claude interactúe directamente con los datos astronómicos reales —sin necesidad de copiar SQL ni exportar archivos— usando lenguaje natural como interfaz principal.

El MCP se consume exclusivamente desde **Claude Desktop** en esta fase, orientado a uso interno por investigadores y desarrolladores del proyecto.

---

### 6.2 Herramientas (Tools) expuestas por el MCP

El servidor MCP define las siguientes herramientas que Claude puede invocar durante una conversación:

####  Grupo 1 — Consultas RAG en lenguaje natural

| Tool | Descripción | Inputs principales |
|---|---|---|
| `rag_query` | Recibe una pregunta en lenguaje natural, la vectoriza, recupera los chunks más similares desde `Embedding_Texto` y devuelve el contexto para que Claude genere una respuesta | `texto_pregunta: str`, `top_k: int = 5` |
| `get_context_for_object` | Recupera documentos y descripciones científicas asociadas a un objeto astronómico específico | `id_objeto: int` o `nombre: str` |

####  Grupo 2 — Gestión de objetos astronómicos (CRUD)

| Tool | Descripción | Inputs principales |
|---|---|---|
| `create_objeto_astronomico` | Registra un nuevo cuerpo celeste (galaxia, estrella, planeta, luna) con su descripción científica | `nombre: str`, `tipo: enum`, `descripcion_cientifica: str`, `atributos: dict` |
| `get_objeto_astronomico` | Consulta un objeto por ID o nombre, retornando su jerarquía completa | `id_objeto: int` o `nombre: str` |
| `update_objeto_astronomico` | Actualiza atributos físicos o descripción de un objeto existente | `id_objeto: int`, `campos: dict` |
| `delete_objeto_astronomico` | Elimina un objeto y sus dependencias en cascada | `id_objeto: int` |
| `list_planetas_habitables` | Lista planetas filtrados por rango de puntaje de habitabilidad y características ambientales | `puntaje_min: float`, `caracteristicas: list` |

####  Grupo 3 — Búsqueda vectorial / semántica

| Tool | Descripción | Inputs principales |
|---|---|---|
| `semantic_search_documentos` | Vectoriza una consulta y busca los documentos más relevantes por similitud coseno en `Embedding_Texto` | `query: str`, `top_k: int`, `estrategia_chunking: str` |
| `semantic_search_imagenes` | Busca imágenes astronómicas similares a una descripción textual usando `Embedding_Imagen` | `query: str`, `top_k: int` |
| `find_similar_planets` | Dado un planeta de referencia, encuentra planetas con descripción científica similar en el espacio vectorial | `id_planeta: int`, `top_k: int` |

####  Grupo 4 — Evaluación de resultados (RAGAS)

| Tool | Descripción | Inputs principales |
|---|---|---|
| `evaluate_rag_response` | Registra y calcula métricas RAGAS (faithfulness, answer_relevancy, context_recall) para una consulta ya resuelta | `id_consulta: int`, `respuesta_generada: str`, `contexto_recuperado: list`, `modelo_eval: str` |
| `get_evaluacion_historica` | Retorna el historial de evaluaciones de un usuario o por rango de fechas para análisis de calidad | `id_usuario: int`, `fecha_desde: date`, `fecha_hasta: date` |

---

### 6.3 Flujo de una interacción típica vía MCP

```
Usuario (Claude Desktop)
    │
    │  "¿Qué planetas tienen condiciones similares a la Tierra?"
    ▼
Claude (LLM)
    │
    ├─► [Tool call] rag_query(texto_pregunta="...", top_k=5)
    │         │
    │         ▼
    │    Servidor MCP
    │         │
    │         ├─ Vectoriza la pregunta → Embedding_Consulta
    │         ├─ Busca similitud coseno → Embedding_Texto / Embedding_Imagen
    │         ├─ Recupera Documento, Objeto_Astronomico, Caracteristica_Ambiental
    │         └─ Retorna contexto estructurado a Claude
    │
    ├─► Claude genera respuesta con el contexto recuperado
    │
    └─► [Tool call] evaluate_rag_response(id_consulta, respuesta, contexto, modelo)
              │
              └─ Guarda métricas en tabla Evaluacion (faithfulness, answer_relevancy, context_recall)
```

---

### 6.4 Arquitectura del servidor MCP

- **Protocolo:** MCP sobre `stdio` (transporte estándar para Claude Desktop)
- **Lenguaje de implementación previsto:** Python con `mcp` SDK oficial de Anthropic
- **Conexión a BD:** `asyncpg` para PostgreSQL + `pgvector` para operaciones vectoriales
- **Modelo de embeddings:** `all-MiniLM-L6-v2` (384d) para texto, `CLIP` (512d) para imágenes
- **Registro en Claude Desktop:** entrada en `claude_desktop_config.json` apuntando al ejecutable del servidor

**Estructura de archivos prevista:**
```
astrodata-mcp/
├── server.py              # Punto de entrada MCP, registro de tools
├── tools/
│   ├── rag.py             # rag_query, get_context_for_object
│   ├── crud.py            # create/get/update/delete objeto astronómico
│   ├── search.py          # semantic_search_documentos, imagenes, find_similar_planets
│   └── evaluacion.py      # evaluate_rag_response, get_evaluacion_historica
├── db/
│   ├── connection.py      # Pool asyncpg
│   └── queries.py         # Queries SQL y vectoriales reutilizables
└── embeddings/
    └── encoder.py         # Lógica de vectorización (MiniLM + CLIP)
```

---

### 6.5 Restricciones y consideraciones de diseño

- **Solo lectura por defecto en RAG:** las tools del Grupo 1 no modifican la BD; solo consultan y registran la consulta del usuario.
- **Autenticación:** el acceso al MCP está restringido a sesiones de Claude Desktop autenticadas; no hay endpoint público.
- **Consistencia híbrida:** al crear o actualizar un `Objeto_Astronomico` vía CRUD, el servidor MCP es responsable de disparar la regeneración del embedding correspondiente en `Embedding_Texto`, manteniendo coherencia entre la capa relacional y la vectorial.
- **Chunking configurable:** `semantic_search_documentos` acepta el parámetro `estrategia_chunking` (`fixed` | `sentence` | `semantic`) para permitir comparar estrategias en tiempo de consulta, alineado con el experimento de la Sección 5.

---
