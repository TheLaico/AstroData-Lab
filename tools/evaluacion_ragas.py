"""Adaptadores MCP de evaluacion RAGAS simplificada."""

from datetime import date
from typing import Any, Dict, Iterable, List, Optional

from mcp.types import Tool

from database.repositorio_consultas import RepositorioConsultas
from database.repositorio_evaluaciones import RepositorioEvaluaciones
from services.evaluation_service import EvaluacionRAGASService


class ToolsEvaluacionRAGAS:
    """Adaptador MCP para registrar y consultar evaluaciones RAGAS."""

    def __init__(self) -> None:
        self.service = EvaluacionRAGASService(
            repo_evaluaciones=RepositorioEvaluaciones(),
            repo_consultas=RepositorioConsultas(),
        )

    def obtener_definiciones_tools(self) -> List[Tool]:
        return [
            Tool(
                name="evaluar_respuesta_rag",
                description="Evalua una respuesta RAG con metricas simplificadas.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_consulta": {"type": ["integer", "string"]},
                        "respuesta_generada": {"type": "string"},
                        "contexto_recuperado": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["id_consulta", "respuesta_generada", "contexto_recuperado"],
                },
            ),
            Tool(
                name="obtener_historial_evaluaciones",
                description="Obtiene historial de evaluaciones RAGAS de un usuario.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_usuario": {"type": ["integer", "string"]},
                        "fecha_desde": {"type": ["string", "null"]},
                        "fecha_hasta": {"type": ["string", "null"]},
                    },
                    "required": ["id_usuario"],
                },
            ),
        ]

    async def evaluar_respuesta_rag(
        self,
        id_consulta: int,
        respuesta_generada: str,
        contexto_recuperado: Iterable[str],
        modelo_eval: str = "ragas-simplificado-mvp",
    ) -> Dict[str, Any]:
        return await self.service.evaluar_respuesta_rag(
            id_consulta=id_consulta,
            respuesta_generada=respuesta_generada,
            contexto_recuperado=contexto_recuperado,
            modelo_eval=modelo_eval,
        )

    async def obtener_historial_evaluaciones(
        self,
        id_usuario: int,
        fecha_desde: Optional[str | date] = None,
        fecha_hasta: Optional[str | date] = None,
    ) -> Dict[str, Any]:
        return await self.service.obtener_historial_evaluaciones(
            id_usuario=id_usuario,
            fecha_desde=fecha_desde,
            fecha_hasta=fecha_hasta,
        )
