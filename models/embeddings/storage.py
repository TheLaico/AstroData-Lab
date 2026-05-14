"""
Módulo para almacenamiento de embeddings en pgvector.

Implementa store_embedding() para persistir:
- Embedding_Texto
- Embedding_Imagen
- Embedding_Consulta
"""


def store_embedding(embedding, embedding_type, metadata=None):
    """
    Almacena embedding en pgvector.
    
    Args:
        embedding: vector de embedding
        embedding_type: "texto", "imagen" o "consulta"
        metadata: dict con información adicional
    
    Returns:
        int: ID del embedding almacenado
    """
    pass


__all__ = ["store_embedding"]
