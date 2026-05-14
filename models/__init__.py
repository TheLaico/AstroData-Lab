"""
Paquete Models - Reestructurado en 6 módulos funcionales.

Módulos:
- sql_queries: Consultas SQL estructuradas
- embeddings: Generación y gestión de embeddings
- semantic_search: Búsqueda semántica con pgvector
- rag_generation: Pipeline RAG completo
- ragas_evaluation: Evaluación de calidad RAGAS
- domain: Modelos de entidades del negocio
"""

from . import sql_queries
from . import embeddings
from . import semantic_search
from . import rag_generation
from . import ragas_evaluation
from . import domain

__all__ = [
    "sql_queries",
    "embeddings",
    "semantic_search",
    "rag_generation",
    "ragas_evaluation",
    "domain",
]
