# AstroData Lab

Sistema de consulta astronómica con capacidades RAG (Retrieval-Augmented Generation) conectado a Claude Desktop vía el protocolo MCP (Model Context Protocol). Permite a Claude consultar, crear y analizar objetos astronómicos almacenados en PostgreSQL con búsqueda vectorial mediante pgvector.

---

## Descripción del proyecto

AstroData Lab resuelve el problema de acceder a grandes volúmenes de datos científicos astronómicos desde un asistente de IA. En lugar de que Claude responda con conocimiento de entrenamiento, puede consultar en tiempo real una base de datos relacional enriquecida con embeddings vectoriales, permitiendo respuestas factuales y actualizadas.

**Tecnologías principales:**

- **PostgreSQL 15+** con extensión **pgvector** para almacenamiento relacional y búsqueda vectorial por similitud coseno.
- **MCP (Model Context Protocol)** para exponer herramientas a Claude Desktop como funciones invocables.
- **Claude Desktop** como interfaz conversacional que consume las herramientas MCP.
- **sentence-transformers / MiniLM** (`all-MiniLM-L6-v2`) para embeddings de texto de 384 dimensiones.
- **CLIP** (`openai/clip-vit-base-patch32`) para embeddings de imágenes astronómicas de 512 dimensiones.
- **RAGAS** como framework de métricas para evaluar la calidad de respuestas RAG (faithfulness, answer relevancy, context recall).

---

## Arquitectura

El proyecto se organiza en capas con responsabilidades bien delimitadas:

| Carpeta | Responsabilidad |
|---|---|
| `server/` | Punto de entrada del servidor MCP. Registra las herramientas e inicia el servidor que Claude Desktop consume. |
| `tools/` | Lógica de negocio expuesta como herramientas MCP. Orquesta flujos RAG, CRUD, búsqueda semántica y evaluación sin implementar persistencia directa. |
| `services/` | Casos de uso de aplicacion. Orquesta RAG, CRUD, busqueda semantica, observaciones y evaluacion sin acoplarse al protocolo MCP. |
| `database/` | Capa de acceso a datos. Repositorios que encapsulan todas las consultas SQL y operaciones con asyncpg y pgvector. |
| `embeddings/` | Codificadores de embeddings. Define la interfaz `CodificadorBase` e implementaciones concretas para texto (MiniLM) e imagen (CLIP). |
| `models/` | Modelos de datos Pydantic v2. Representan las entidades del dominio (objetos astronómicos, consultas, evaluaciones, resultados). |
| `config/` | Configuración centralizada. Lee variables de entorno desde `.env` y las expone como objeto de ajustes tipado. |
| `tests/` | Suite de pruebas con pytest y mocks. Cubre RAG, CRUD, búsqueda semántica y evaluación RAGAS sin necesidad de BD real. |

---

## Requisitos previos

- Python 3.11 o superior
- PostgreSQL 15 o superior con la extensión `pgvector` instalada y activa
- Claude Desktop instalado en macOS o Linux

---

## Instalación paso a paso

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/astrodata-mcp.git
cd astrodata-mcp

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requerimientos.txt

# 4. Copiar archivo de variables de entorno
cp config/.env.ejemplo config/.env

# 5. Editar config/.env con tus credenciales reales
#    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
```

---

## Configuración de la base de datos

Ejecutar los scripts SQL en el siguiente orden desde una conexión a PostgreSQL:

```bash
# 1. Habilitar la extensión pgvector y crear el esquema base
psql -U tu_usuario -d tu_base -f sql/pgvector.sql

# 2. Crear todas las tablas del dominio astronómico
psql -U tu_usuario -d tu_base -f sql/PostgreSQL.sql
```

> Es importante respetar el orden: pgvector debe habilitarse antes de crear tablas con columnas de tipo `vector`.

---

## Conexión a Claude Desktop

1. Abrir (o crear) el archivo de configuración de Claude Desktop:
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Linux:** `~/.config/Claude/claude_desktop_config.json`

2. Editar el archivo reemplazando la ruta del proyecto:

```json
{
  "mcpServers": {
    "astrodata-mcp": {
      "command": "python",
      "args": ["/ruta/real/astrodata-mcp/server/servidor_mcp.py"],
      "env": {
        "PYTHONPATH": "/ruta/real/astrodata-mcp"
      }
    }
  }
}
```

3. Guardar el archivo y **reiniciar Claude Desktop** completamente.

4. Verificar que el ícono de herramientas aparezca activo en la interfaz de Claude.

---

## Uso básico

Una vez conectado el servidor MCP, puedes hacer preguntas directamente a Claude:

**Consulta RAG sobre un objeto:**
> "¿Cuáles son las características científicas de la nebulosa de Orión según la base de datos?"

**Búsqueda de planetas habitables:**
> "Lista los planetas con puntaje de habitabilidad mayor a 0.7 que tenemos registrados."

**Evaluación de una respuesta:**
> "Evalúa qué tan buena fue la respuesta que generaste para la consulta número 42."

---

## Ejecución de tests

```bash
# Ejecutar toda la suite con salida detallada
pytest tests/ -v

# Ejecutar con reporte de cobertura
pytest tests/ -v --cov=tools --cov=database --cov-report=term-missing

# Ejecutar solo un módulo de tests
pytest tests/prueba_rag.py -v
```

> Los tests usan mocks completos de repositorios y codificadores, por lo que **no requieren una base de datos activa** para ejecutarse.

---

## Estado MVP estable

Esta version incluye una base ejecutable para la entrega universitaria:

- Servidor MCP con tools reales para RAG, CRUD astronomico, busqueda semantica y evaluacion RAGAS simplificada.
- Scripts SQL versionados en `sql/`: `001_schema_relacional.sql`, `002_pgvector.sql` y `003_seed_data.sql`.
- Configuracion de pruebas en `pytest.ini`; `pytest tests/` descubre `prueba_*.py`.
- Plantilla local `config/.env.example`; copiarla a `config/.env` y ajustar credenciales.

Orden recomendado para recrear la base:

```bash
psql -d astrodata -f sql/001_schema_relacional.sql
psql -d astrodata -f sql/002_pgvector.sql
psql -d astrodata -f sql/003_seed_data.sql
```

Para PostgreSQL local usar `DB_SSL=disable`. Para Neon u otros servicios con SSL obligatorio usar `DB_SSL=require`.

Se recomienda Python 3.11 o 3.12 para evitar incompatibilidades de `torch` y `sentence-transformers`.

---

## Principios de diseño

| Principio SOLID | Módulo que lo implementa | Por qué |
|---|---|---|
| **SRP** — Responsabilidad Única | `tools/consulta_rag.py`, `tools/gestion_objetos.py` | Cada clase de tools solo orquesta su flujo específico; no implementa persistencia ni cálculo de embeddings. |
| **OCP** — Abierto/Cerrado | `database/repositorio_objetos.py`, `database/repositorio_documentos.py` | Los repositorios pueden extenderse con nuevos métodos sin modificar los existentes. |
| **LSP** — Sustitución de Liskov | `embeddings/codificador_texto.py`, `embeddings/codificador_imagen.py` | Ambas implementaciones son intercambiables donde se use `CodificadorBase` sin romper el comportamiento esperado. |
| **DIP** — Inversión de Dependencias | `embeddings/interfaz_codificador.py` + inyección en `tools/` | Las herramientas dependen de la abstracción `CodificadorBase`, nunca de `CodificadorTexto` o `CodificadorImagen` directamente. |

---

## Referencias

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997.
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
- Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217.
