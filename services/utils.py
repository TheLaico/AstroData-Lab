"""Utilidades compartidas por servicios de aplicacion."""

from typing import Any, Dict, Optional


def dump_model(obj: Any) -> Dict[str, Any]:
    """Convierte modelos Pydantic, objetos simples o mocks en diccionarios."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        try:
            volcado = obj.model_dump()
            if isinstance(volcado, dict):
                return volcado
        except Exception:
            pass
    datos = {}
    for nombre in (
        "id_objeto", "nombre", "descripcion_cientifica", "id_doc", "titulo",
        "id_imagen", "ruta_archivo", "id_telescopio", "tipo", "ubicacion",
        "id_observacion", "id_planeta", "fecha", "descripcion",
    ):
        if hasattr(obj, nombre):
            valor = getattr(obj, nombre)
            if not callable(valor):
                datos[nombre] = valor
    if datos:
        return datos
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"valor": obj}


def to_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} debe ser un entero positivo.") from exc


def to_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    return to_int(value, field_name)
