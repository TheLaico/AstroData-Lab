"""Tools de evaluacion RAGAS simplificada para el MVP."""

from datetime import date
import re
from typing import Any, Dict, Iterable, List, Optional

from mcp.types import Tool

from database.repositorio_consultas import RepositorioConsultas
from database.repositorio_evaluaciones import RepositorioEvaluaciones
from models.evaluacion_ragas_entrada_model import EvaluacionRAGASEntrada


class ToolsEvaluacionRAGAS:
    """Adaptador MCP para registrar y consultar evaluaciones RAGAS."""

    def __init__(self) -> None:
        self.repo_evaluaciones = RepositorioEvaluaciones()
        self.repo_consultas = RepositorioConsultas()

    def obtener_definiciones_tools(self) -> List[Tool]:
        return [
            Tool(
                name="evaluar_respuesta_rag",
                description="Evalua una respuesta RAG con metricas simplificadas.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_consulta": {"type": "integer"},
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
                        "id_usuario": {"type": "integer"},
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
        if not isinstance(id_consulta, int) or id_consulta <= 0:
            return {"error": "id_consulta debe ser un entero positivo."}

        consulta = await self.repo_consultas.obtener_consulta_por_id(id_consulta)
        if consulta is None:
            return {"error": "No existe una consulta con el id indicado."}

        contexto = " ".join(str(c) for c in contexto_recuperado or [])
        faithfulness = self._solapamiento(respuesta_generada, contexto)
        answer_relevancy = self._solapamiento(
            respuesta_generada,
            getattr(consulta, "texto_pregunta", "") or "",
        )
        context_recall = self._solapamiento(contexto, respuesta_generada)

        entrada = EvaluacionRAGASEntrada(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_recall=context_recall,
            modelo_eval=modelo_eval,
            id_consulta=id_consulta,
        )
        evaluacion = await self.repo_evaluaciones.registrar_evaluacion_ragas(entrada)
        return self._evaluacion_a_respuesta(evaluacion)

    async def obtener_historial_evaluaciones(
        self,
        id_usuario: int,
        fecha_desde: Optional[str | date] = None,
        fecha_hasta: Optional[str | date] = None,
    ) -> Dict[str, Any]:
        if not isinstance(id_usuario, int) or id_usuario <= 0:
            return {"error": "id_usuario debe ser un entero positivo."}

        desde = self._parse_fecha(fecha_desde)
        hasta = self._parse_fecha(fecha_hasta)
        evaluaciones = await self.repo_evaluaciones.listar_evaluaciones_por_usuario(
            id_usuario,
            desde,
            hasta,
        )
        resumen = None
        try:
            resumen_obj = await self.repo_evaluaciones.calcular_resumen_usuario(id_usuario)
            resumen = {
                "promedio_general": float(resumen_obj.promedio_metricas),
                "calidad": resumen_obj.calidad,
            }
        except Exception:
            resumen = {"promedio_general": 0.0, "calidad": "baja"}

        return {
            "id_usuario": id_usuario,
            "evaluaciones": [self._evaluacion_a_respuesta(e) for e in evaluaciones],
            "resumen_usuario": resumen,
        }

    def _evaluacion_a_respuesta(self, evaluacion: Any) -> Dict[str, Any]:
        promedio = (
            float(evaluacion.faithfulness)
            + float(evaluacion.answer_relevancy)
            + float(evaluacion.context_recall)
        ) / 3.0
        return {
            "id_evaluacion": evaluacion.id_evaluacion,
            "id_consulta": evaluacion.id_consulta,
            "faithfulness": float(evaluacion.faithfulness),
            "answer_relevancy": float(evaluacion.answer_relevancy),
            "context_recall": float(evaluacion.context_recall),
            "modelo_eval": evaluacion.modelo_eval,
            "fecha": evaluacion.fecha.isoformat() if hasattr(evaluacion.fecha, "isoformat") else str(evaluacion.fecha),
            "promedio_metricas": promedio,
            "calidad": self._calidad(promedio),
        }

    def _calidad(self, promedio: float) -> str:
        if promedio >= 0.7:
            return "alta"
        if promedio >= 0.5:
            return "media"
        return "baja"

    def _parse_fecha(self, valor: Optional[str | date]) -> Optional[date]:
        if valor is None or isinstance(valor, date):
            return valor
        return date.fromisoformat(valor)

    def _solapamiento(self, texto_a: str, texto_b: str) -> float:
        tokens_a = self._tokens(texto_a)
        tokens_b = self._tokens(texto_b)
        if not tokens_a or not tokens_b:
            return 0.0
        return min(1.0, len(tokens_a & tokens_b) / len(tokens_a))

    def _tokens(self, texto: str) -> set[str]:
        stopwords = {
            "el", "la", "los", "las", "un", "una", "de", "del", "y", "o",
            "en", "con", "por", "para", "que", "es", "son", "a",
        }
        return {
            t
            for t in re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ0-9]+", texto.lower())
            if len(t) > 2 and t not in stopwords
        }
