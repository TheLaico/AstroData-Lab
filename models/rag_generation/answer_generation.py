"""
Módulo de generación de respuestas.

Implementa generate_answer() para crear respuestas contextualizadas
usando LLM con el contexto recuperado.
"""


def generate_answer(query, context, model="gpt-3.5-turbo"):
    """
    Genera respuesta contextualizada usando LLM.
    
    Args:
        query: string con la pregunta del usuario
        context: string con contexto recuperado por RAG
        model: modelo LLM a usar (default gpt-3.5-turbo)
    
    Returns:
        str: Respuesta generada por el LLM
    """
    pass


__all__ = ["generate_answer"]
