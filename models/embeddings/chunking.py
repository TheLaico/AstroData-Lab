"""
Módulo para chunking de documentos.

Implementa dos estrategias:
- Fixed: chunks de 256 tokens
- Sentence-based: chunks de 128-512 tokens basados en oraciones
"""


def chunk_document(document, strategy="fixed", chunk_size=256):
    """
    Divide un documento en chunks para embeddings.
    
    Args:
        document: string con contenido a chunking
        strategy: "fixed" o "sentence_based"
        chunk_size: tamaño del chunk en tokens (default 256)
    
    Returns:
        list: Lista de chunks
    """
    pass


__all__ = ["chunk_document"]
