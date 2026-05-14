"""
Módulo de almacenamiento de resultados.

Implementa save_result() para persistir:
- Respuesta generada
- Relevancia y métricas
- Metadata de la consulta
"""


def save_result(query, answer, context, relevance_score):
    """
    Almacena resultado en tabla Resultado.
    
    Args:
        query: string con consulta original
        answer: string con respuesta generada
        context: string con contexto utilizado
        relevance_score: float con score de relevancia
    
    Returns:
        int: ID del resultado almacenado
    """
    pass


__all__ = ["save_result"]
