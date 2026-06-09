"""
Conjunto fijo de consultas de prueba y ground truth para evaluar el sistema RAG.

CONSULTAS_PRUEBA: 10 preguntas representativas del dominio astronómico.
GROUND_TRUTH: asocia cada consulta con los id_doc esperados como relevantes.
    - Actualizar los id_doc cuando se carguen documentos reales en la BD.
    - Un resultado se considera correcto (Context Recall = 1) si todos los
      id_doc del ground truth aparecen entre los chunks recuperados.

Uso en evaluación:
    from evaluation.consultas_prueba import CONSULTAS_PRUEBA, GROUND_TRUTH

    for pregunta in CONSULTAS_PRUEBA:
        resultado = await rag_service.rag_query(pregunta, top_k=5)
        ids_recuperados = {c["id_doc"] for c in resultado["chunks_recuperados"]}
        ids_esperados   = set(GROUND_TRUTH.get(pregunta, []))
        context_recall  = len(ids_esperados & ids_recuperados) / len(ids_esperados) if ids_esperados else 0.0
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# 10 consultas de prueba fijas
# ---------------------------------------------------------------------------

CONSULTAS_PRUEBA: List[str] = [
    "planetas con condiciones similares a la Tierra",
    "objetos con posible habitabilidad",
    "planetas con atmósfera densa",
    "lunas con posible océano interno",
    "estrellas similares al Sol",
    "planetas rocosos cercanos a su estrella",
    "cuerpos celestes con baja temperatura superficial",
    "objetos observados por telescopios espaciales",
    "planetas con evidencia de agua líquida",
    "sistemas estelares con múltiples planetas",
]

# ---------------------------------------------------------------------------
# Ground truth: consulta → lista de id_doc relevantes esperados
# ---------------------------------------------------------------------------
# IMPORTANTE: estos id_doc son marcadores de posición.
# Reemplazar con los ids reales una vez que los documentos estén cargados
# en la base de datos. El sistema de evaluación usa esta tabla para calcular
# Context Recall en evaluar_respuesta_rag().

GROUND_TRUTH: Dict[str, List[int]] = {
    "planetas con condiciones similares a la Tierra":     [],  # TODO: [1, 4, 7]
    "objetos con posible habitabilidad":                  [],  # TODO: [2, 5, 8]
    "planetas con atmósfera densa":                       [],  # TODO: [3, 6]
    "lunas con posible océano interno":                   [],  # TODO: [9, 12]
    "estrellas similares al Sol":                         [],  # TODO: [10, 13]
    "planetas rocosos cercanos a su estrella":            [],  # TODO: [4, 11]
    "cuerpos celestes con baja temperatura superficial":  [],  # TODO: [14, 15]
    "objetos observados por telescopios espaciales":      [],  # TODO: [1, 2, 3]
    "planetas con evidencia de agua líquida":             [],  # TODO: [5, 7, 9]
    "sistemas estelares con múltiples planetas":          [],  # TODO: [6, 8, 10]
}


def calcular_context_recall(pregunta: str, ids_recuperados: set) -> float:
    """
    Calcula el Context Recall para una consulta dada.

    Context Recall = |ids_esperados ∩ ids_recuperados| / |ids_esperados|

    Retorna 0.0 si la consulta no tiene ground truth definido o si
    los ids esperados aún son marcadores vacíos.

    Args:
        pregunta: Texto exacto de la consulta (debe coincidir con GROUND_TRUTH).
        ids_recuperados: Conjunto de id_doc devueltos por el sistema RAG.

    Returns:
        Float entre 0.0 y 1.0.
    """
    ids_esperados = set(GROUND_TRUTH.get(pregunta, []))
    if not ids_esperados:
        return 0.0
    return len(ids_esperados & ids_recuperados) / len(ids_esperados)
