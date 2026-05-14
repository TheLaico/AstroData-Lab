"""
Módulo para comparación de estrategias de chunking.

Implementa compare_chunking() para experimentar:
- Fixed chunks (256 tokens)
- Sentence-based chunks (128-512 tokens)
- Compara métricas RAGAS entre estrategias
"""


def compare_chunking(documents, test_queries):
    """
    Compara rendimiento entre Fixed y Sentence-based chunking.
    
    Args:
        documents: list de documentos para chunking
        test_queries: list de consultas de prueba
    
    Returns:
        dict con resultados de ambas estrategias:
            - fixed: métricas RAGAS
            - sentence_based: métricas RAGAS
    """
    pass


__all__ = ["compare_chunking"]
