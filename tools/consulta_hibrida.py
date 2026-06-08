"""Tool MCP para consulta híbrida de documentos astronómicos."""

from typing import Any, Dict, List, Optional

from mcp.types import Tool

from services.servicio_consulta_hibrida import ServicioConsultaHibrida


class ToolsConsultaHibrida:
    """Adaptador MCP para la consulta híbrida de AstroData Lab."""

    def __init__(self, codificador: Any, codificador_imagen: Any = None) -> None:
        self.service = ServicioConsultaHibrida(
            codificador=codificador,
            codificador_imagen=codificador_imagen,
        )

    def obtener_definiciones_tools(self) -> List[Tool]:
        return [
            Tool(
                name="consulta_hibrida",
                description=(
                    "Ejecuta una consulta híbrida combinando RAG con filtros exactos "
                    "sobre metadatos de documentos astronómicos."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texto_pregunta": {"type": "string"},
                        "filtros": {
                            "type": ["object", "null"],
                            "description": "Filtros exactos por metadatos del documento.",
                        },
                        "top_k": {"type": ["integer", "string"], "default": 5},
                        "alpha": {"type": ["number", "string"], "default": 0.7},
                    },
                    "required": ["texto_pregunta"],
                },
            )
        ]

    async def consulta_hibrida(
        self,
        texto_pregunta: str,
        filtros: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        alpha: float = 0.7,
    ) -> Dict[str, Any]:
        return await self.service.consulta_hibrida(
            texto_pregunta=texto_pregunta,
            filtros=filtros,
            top_k=top_k,
            alpha=alpha,
        )
