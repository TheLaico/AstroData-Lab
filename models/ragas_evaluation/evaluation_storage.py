"""
Módulo para almacenamiento de evaluaciones.

Implementa save_evaluation() para persistir:
- Métricas RAGAS en tabla Evaluacion
- Modelo utilizado
- Fecha y metadata
"""


def save_evaluation(metrics, model_name, chunking_strategy, metadata=None):
    """
    Almacena evaluación en tabla Evaluacion.
    
    Args:
        metrics: dict con métricas RAGAS
        model_name: string con nombre del modelo
        chunking_strategy: "fixed" o "sentence_based"
        metadata: dict adicional
    
    Returns:
        int: ID de la evaluación almacenada
    """
    pass


__all__ = ["save_evaluation"]
