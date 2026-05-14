"""
Módulo de recuperación de contexto.

Implementa retrieve_context() para armar contexto desde:
- Chunks recuperados por similitud
- Objetos relevantes de la base de datos
"""


def retrieve_context(query_embedding, sql_filter=None, top_k=5):
    """
    Recupera contexto relevante para responder una consulta.
    
    Args:
        query_embedding: vector de embedding de la consulta
        sql_filter: dict con filtros SQL opcionales
        top_k: número de chunks a recuperar (default 5)
    
    Returns:
        str: Contexto formateado para pasar al LLM
    """
    pass


__all__ = ["retrieve_context"]
