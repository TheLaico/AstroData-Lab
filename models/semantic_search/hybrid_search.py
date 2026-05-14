"""
Módulo de búsqueda híbrida.

Combina búsqueda SQL tradicional con búsqueda semántica:
- Filtrar por tipo + similitud semántica
- Fusionar resultados SQL y vectoriales
"""


def hybrid_search(sql_filter, query_embedding, top_k=10):
    """
    Ejecuta búsqueda híbrida: SQL + semántica.
    
    Args:
        sql_filter: dict con criterios SQL
        query_embedding: vector de embedding
        top_k: número de resultados (default 10)
    
    Returns:
        list: Resultados ordenados por relevancia combinada
    """
    pass


__all__ = ["hybrid_search"]
