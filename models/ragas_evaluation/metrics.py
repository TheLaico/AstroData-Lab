"""
Módulo de métricas RAGAS.

Implementa evaluate_ragas() para calcular:
- Faithfulness: coherencia entre respuesta y contexto
- Answer relevancy: relevancia entre respuesta y pregunta
- Context recall: cobertura de información relevante
"""


def evaluate_ragas(question, answer, context):
    """
    Calcula métricas RAGAS para evaluar calidad de RAG.
    
    Args:
        question: str con la pregunta original
        answer: str con la respuesta generada
        context: str con el contexto utilizado
    
    Returns:
        dict con métricas:
            - faithfulness: float 0-1
            - answer_relevancy: float 0-1
            - context_recall: float 0-1
    """
    pass


__all__ = ["evaluate_ragas"]
