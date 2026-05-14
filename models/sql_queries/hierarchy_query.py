"""
Módulo para navegación de jerarquía astronómica.

Permite navegar: galaxia → sistema → estrella → planeta → luna
Implementa get_hierarchy() para recuperar la estructura completa de un objeto.
"""


def get_hierarchy(obj_id, obj_type):
    """
    Recupera la jerarquía completa de un objeto astronómico.
    
    Args:
        obj_id: ID del objeto astronómico
        obj_type: Tipo de objeto (galaxia, sistema, estrella, planeta, luna)
    
    Returns:
        dict: Estructura jerárquica completa del objeto
    """
    pass


__all__ = ["get_hierarchy"]
