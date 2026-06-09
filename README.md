# AstroData Lab

Sistema híbrido de exploración astronómica que combina un modelo relacional jerárquico con búsqueda vectorial semántica, expuesto como servidor MCP (Model Context Protocol) para Claude Desktop. Permite consultar, analizar y razonar sobre objetos astronómicos almacenados en PostgreSQL, usando un pipeline RAG completo con evaluación de calidad.

---

## Descripción del proyecto

AstroData Lab resuelve el problema de coexistencia entre datos estructurados y datos semánticos sin un modelo unificado. Los sistemas tradicionales permiten filtros exactos (temperatura < 300K, masa > 0.5 M☉), pero no pueden responder preguntas como *"planetas con condiciones similares a la Tierra"* porque esa noción es semántica y multivariada: no se reduce a umbrales numéricos. El componente vectorial de este sistema cubre exactamente ese espacio.

**Tecnologías principales:**

| Tecnología | Rol en el sistema |
|---|---|
| **PostgreSQL 15+** + **pgvector** | Almacenamiento relacional y búsqueda vectorial por distancia coseno (`<=>`) |
| **MCP (Model Context Protocol)** | Expone las herramientas a Claude Desktop como funciones invocables |
| **sentence-transformers / MiniLM** (`all-MiniLM-L6-v2`) | Embeddings de texto — 384 dimensiones |
| **CLIP** (`openai/clip-vit-base-patch32`) | Embeddings de imágenes astronómicas — 512 dimensiones |
| **RAGAS simplificado** | Métricas de evaluación RAG: faithfulness, answer relevancy, context recall |

Los modelos de texto e imagen son deliberadamente separados: MiniLM opera en espacio semántico de lenguaje natural mientras que CLIP opera en espacio multimodal texto-imagen. Usar el mismo modelo para ambas modalidades degradaría la calidad de las representaciones.

---

## Arquitectura

```
Claude Desktop
      │  MCP protocol
      ▼
server/servidor_mcp.py        ← punto de entrada, registra tools e inicia servidor
      │
      ▼
tools/                        ← herramientas MCP (adaptadores de protocolo)
  ├── consulta_rag.py
  ├── consulta_hibrida.py     ← combina vector search + filtros relacionales en SQL
  ├── busqueda_semantica.py
  ├── gestion_objetos.py
  ├── evaluacion_ragas.py
  ├── presentacion.py
  ├── modo_profesor.py
  └── terminal_profesor.py
      │
      ▼
services/                     ← casos de uso de aplicación
  ├── rag_service.py
  ├── servicio_consulta_hibrida.py
  ├── semantic_search_service.py
  ├── objetos_service.py
  ├── chunking_service.py     ← estrategias: fixed, sentence, semantic
  └── evaluation_service.py
      │
      ▼
database/                     ← repositorios (capa de acceso a datos)
  ├── repositorio_documentos.py   ← incluye consulta_hibrida_sql() con query builder
  ├── repositorio_consultas.py
  ├── repositorio_objetos.py
  ├── repositorio_observaciones.py
  └── embeddings/
        ├── codificador_texto.py  ← MiniLM, async con run_in_executor
        └── codificador_imagen.py ← CLIP, async con run_in_executor
      │
      ▼
models/                       ← entidades Pydantic v2
evaluation/
  └── consultas_prueba.py     ← 10 consultas fijas + ground truth + calcular_context_recall()
```

---

## Modelo de datos

### Jerarquía IS-A de objetos astronómicos

La entidad central es `Objeto_Astronomico`, que actúa como superentidad de la jerarquía:

```
Objeto_Astronomico (id_objeto PK, nombre, descripcion_cientifica)
    ├── Galaxia          (id_objeto PK + FK → Objeto_Astronomico ON DELETE CASCADE)
    ├── Sistema_Estelar  (id_objeto PK + FK → Objeto_Astronomico ON DELETE CASCADE)
    ├── Estrella         (id_objeto PK + FK → Objeto_Astronomico ON DELETE CASCADE)
    ├── Planeta          (id_objeto PK + FK → Objeto_Astronomico ON DELETE CASCADE)
    └── Luna             (id_objeto PK + FK → Objeto_Astronomico ON DELETE CASCADE)
```

Esta implementación de herencia relacional mediante PK compartida garantiza que no puede existir una fila en `Galaxia` sin su correspondiente en `Objeto_Astronomico`. El `ON DELETE CASCADE` asegura que eliminar la superentidad elimina automáticamente el subtipo, evitando registros huérfanos en los subtipos.

> **Nota de diseño:** puede existir un `Objeto_Astronomico` sin subtipo correspondiente (registro base sin especialización). Esto es una limitación conocida de la implementación IS-A en SQL relacional puro; la integridad inversa —que todo objeto base tenga exactamente un subtipo— se delega a la capa de servicio en `objetos_service.py`, donde `crear_objeto_astronomico` usa una transacción compensatoria para garantizar que si la creación del documento/embedding falla tras insertar el objeto base, el objeto base se elimina automáticamente.

### Documentos e imágenes — opcionalidad intencional

- `Documento.id_objeto` es una FK opcional (permite NULL). Esto es **deliberado**: el sistema admite documentos científicos generales sobre astronomía que no están referenciados a un cuerpo celeste específico (artículos de revisión, manuales de telescopios, etc.).

- `Imagen.id_doc` es una FK opcional (permite NULL). Esto es **deliberado**: el sistema admite imágenes de referencia general (mapas estelares, diagramas) que no pertenecen a un documento específico. Para que estas imágenes tengan sentido semántico en el sistema deben tener al menos `descripcion` o `etiquetas` no nulas, de modo que puedan generar embeddings vectoriales y ser recuperables por búsqueda semántica.

### Integridad de Resultado

La tabla `Resultado` tiene dos FKs opcionales: `id_doc` e `id_imagen`. El sistema multimodal permite resultados que apuntan solo a un documento, solo a una imagen, o a ambos. La restricción de que **al menos una debe ser no nula** está implementada como validación en la capa de repositorio (`repositorio_consultas.py`):

```python
if datos.id_doc is None and datos.id_imagen is None:
    raise ValueError("El resultado debe estar asociado a un documento o a una imagen.")
```

Esta validación se evalúa antes de cualquier operación en base de datos, evitando registros de resultado sin referencia a contenido recuperado.

---

## Campos vectorizables

Solo se vectorizan contenidos textuales reales. Los identificadores y los vectores ya almacenados no son insumos de vectorización:

| Campo | Tabla | Modelo | Justificación |
|---|---|---|---|
| `descripcion_cientifica` | `Objeto_Astronomico` | MiniLM | Permite búsquedas semánticas como *"planeta con condiciones similares a la Tierra"* que no son resolubles con filtros numéricos exactos |
| `contenido_chunk` | `Embedding_Texto` | MiniLM | Fragmento de texto del documento; se vectoriza el contenido real del chunk, no el `chunk_id` ni el vector resultante |
| `descripcion` | `Imagen` | CLIP texto | Contexto narrativo de la imagen |
| `etiquetas` | `Imagen` | CLIP texto | Terminología técnica astronómica concentrada; permite recuperar imágenes por términos exactos |
| píxeles de imagen | `Imagen` | CLIP imagen | Vector visual; habilita búsqueda cruzada texto→imagen |
| `texto_pregunta` | `Consulta` | MiniLM | Vector de la consulta del usuario para comparación con chunks |

> `chunk_id` es un identificador secuencial; `vector` es el resultado de la vectorización. Ninguno de los dos es un campo vectorizable.

---

## Pipeline RAG

### Consulta híbrida SQL + vector search

La consulta híbrida no ejecuta dos queries separadas en Python para luego fusionarlas. El método `consulta_hibrida_sql()` en `repositorio_documentos.py` construye una única query SQL que combina vector search y filtros relacionales en una sola operación PostgreSQL:

```sql
SELECT et.id_doc, d.titulo, et.chunk_id,
       et.contenido_chunk          AS contenido,
       et.estrategia_chunking,
       1 - (et.vector <=> $1::vector) AS similitud
FROM Embedding_Texto et
JOIN Documento d ON d.id_doc = et.id_doc
WHERE d.idioma = $3          -- filtro relacional (dinámico)
ORDER BY et.vector <=> $1::vector
LIMIT $2
```

Los filtros se construyen dinámicamente mediante un **query builder** con whitelist explícita de columnas permitidas (`idioma`, `fuente`, `id_objeto`), lo que previene inyección SQL al tiempo que permite flexibilidad en los filtros enviados desde el tool MCP.

### Estrategias de chunking

`ChunkingService` implementa tres estrategias reales, todas conectadas al método `dividir()`:

| Estrategia | Descripción | Caso de uso |
|---|---|---|
| `fixed` | Ventanas de 160 palabras con solapamiento de 24 | Textos uniformes, sin estructura de oraciones clara |
| `sentence` | Agrupa oraciones respetando límite de 120 palabras con solapamiento de 1 oración | Descripciones científicas con oraciones semánticamente completas |
| `semantic` | Detecta cambios temáticos por similitud Jaccard entre oraciones contiguas; corta cuando la cohesión cae bajo el umbral | Documentos con secciones temáticas diferenciadas (composición atmosférica, geología, historia orbital) |

La estrategia `semantic` es especialmente adecuada para el corpus de AstroData Lab: una descripción científica de un planeta típicamente tiene párrafos sobre composición química, condiciones de temperatura, historia orbital y posibilidad de habitabilidad. El chunker semántico los separa temáticamente, produciendo chunks más cohesivos que las estrategias de tamaño fijo o por oraciones.

### Evaluación RAG — consultas de prueba y ground truth

El módulo `evaluation/consultas_prueba.py` define el conjunto fijo de evaluación:

```python
CONSULTAS_PRUEBA = [
    "planetas con condiciones similares a la Tierra",
    "objetos con posible habitabilidad",
    "planetas con atmósfera densa",
    "lunas con posible océano interno",
    "estrellas similares al Sol",
    "planetas rocosos cercanos a su estrella",
    "cuerpos celestes con baja temperatura superficial",
    "objetos observados por telescopios espaciales",
    "planetas con evidencia de agua líquida",
    "sistemas estelares con múltiples planetas",
]
```

El `GROUND_TRUTH` asocia cada consulta con los `id_doc` relevantes esperados. La función `calcular_context_recall(pregunta, ids_recuperados)` calcula:

```
Context Recall = |ids_esperados ∩ ids_recuperados| / |ids_esperados|
```

Los `id_doc` del ground truth son marcadores de posición (`TODO`) hasta que los documentos reales sean cargados en la base de datos. Una vez cargados, se actualizan en el archivo y el sistema de evaluación queda listo para medir Context Recall de forma reproducible con las mismas 10 consultas en cada iteración.

---

## Requisitos previos

- Python 3.11 o 3.12 (recomendado para compatibilidad con `torch` y `sentence-transformers`)
- PostgreSQL 15 o superior con la extensión `pgvector` instalada y activa
- Claude Desktop

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TheLaico/AstroData-Lab.git
cd AstroData-Lab

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows (PowerShell)

# 3. Instalar dependencias
pip install -r requerimientos.txt

# 4. Configurar variables de entorno
cp config/.env.example config/.env
# Editar config/.env: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
# Para PostgreSQL local usar DB_SSL=disable
# Para Neon u otros servicios con SSL usar DB_SSL=require
```

---

## Configuración de la base de datos

Ejecutar los scripts en orden estricto (pgvector debe existir antes de crear columnas de tipo `vector`):

```bash
psql -d astrodata -f sql/001_schema_relacional.sql
psql -d astrodata -f sql/002_pgvector.sql
psql -d astrodata -f sql/003_seed_data.sql
```

Los índices están parametrizados por volumen esperado:
- `Embedding_Texto`: IVFFlat con `lists=100` (mayor volumen de chunks de documentos)
- `Embedding_Imagen` y `Embedding_Consulta`: IVFFlat con `lists=50` (menor volumen)

---

## Conexión a Claude Desktop

1. Abrir el archivo de configuración de Claude Desktop:
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Agregar la entrada del servidor:

```json
{
  "mcpServers": {
    "astrodata-mcp": {
      "command": "python",
      "args": ["C:/ruta/real/astrodata-mcp/server/servidor_mcp.py"],
      "env": {
        "PYTHONPATH": "C:/ruta/real/astrodata-mcp"
      }
    }
  }
}
```

3. Guardar y reiniciar Claude Desktop completamente.

---

## Ejecución de tests

```bash
# Suite completa
pytest tests/ -v

# Con cobertura
pytest tests/ -v --cov=tools --cov=database --cov-report=term-missing

# Módulo específico
pytest tests/prueba_rag.py -v
```

Los tests usan mocks completos de repositorios y codificadores. **No requieren base de datos activa.**

---

## Principios de diseño

| Principio SOLID | Módulo | Implementación |
|---|---|---|
| **SRP** | `tools/consulta_rag.py`, `tools/gestion_objetos.py` | Cada tool solo orquesta su flujo; no implementa persistencia ni cálculo de embeddings |
| **OCP** | `database/repositorio_objetos.py`, `database/repositorio_documentos.py` | Extensibles con nuevos métodos sin modificar los existentes |
| **LSP** | `database/embeddings/codificador_texto.py`, `codificador_imagen.py` | Ambas implementaciones son intercambiables donde se use `CodificadorBase` |
| **DIP** | `database/embeddings/interfaz_codificador.py` + inyección en `tools/` | Las herramientas dependen de la abstracción `CodificadorBase`, nunca de la implementación concreta |

### Decisiones técnicas adicionales

- **Embeddings async con `run_in_executor`:** `sentence-transformers` y CLIP ejecutan inferencia síncrona sobre PyTorch. Llamarlos directamente dentro de un método `async` bloquea el event loop del servidor MCP. Ambos codificadores delegan la inferencia a un `ThreadPoolExecutor` mediante `asyncio.run_in_executor`, liberando el loop durante el cómputo.

- **Transacción compensatoria en creación de objetos:** `crear_objeto_astronomico` en `objetos_service.py` crea el objeto base y su documento/embedding en pasos secuenciales. Si el embedding falla tras crear el objeto, la función elimina automáticamente el objeto base para evitar registros huérfanos (patrón de compensación explícita, alternativa pragmática a compartir una conexión de transacción entre repositorios independientes).

- **Separación de modelos por modalidad:** `CodificadorTexto` (MiniLM, 384 dims) y `CodificadorImagen` (CLIP, 512 dims) son clases independientes que implementan la misma interfaz `CodificadorBase`. El servidor MCP los instancia por separado y los inyecta donde corresponde, garantizando que las búsquedas texto→texto y texto→imagen usen el espacio vectorial apropiado para cada modalidad.

---

## Referencias

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997.
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
- Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217.
