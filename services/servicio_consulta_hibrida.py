"""Servicio de consulta híbrida: SQL + vector search en una sola query."""

from typing import Any, Dict, List, Optional

from database.repositorio_documentos import RepositorioDocumentos


class ServicioConsultaHibrida:
    """Delega la fusión SQL+vector al repositorio en lugar de hacerla en Python."""

    def __init__(
        self,
        codificador: Any,
        codificador_imagen: Any = None,
    ) -> None:
        self.codificador = codificador
        self.repo_documentos = RepositorioDocumentos()

    async def consulta_hibrida(
        self,
        texto_pregunta: str,
        filtros: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        alpha: float = 0.7,  # reservado para compatibilidad con el tool MCP
    ) -> Dict[str, Any]:
        if not isinstance(texto_pregunta, str) or not texto_pregunta.strip():
            return {"error": "La pregunta no puede estar vacía."}

        try:
            top_k = int(top_k)
            alpha = float(alpha)
        except (TypeError, ValueError):
            return {"error": "top_k debe ser entero y alpha un número."}

        if top_k <= 0:
            return {"error": "top_k debe ser un entero positivo."}
        if alpha < 0.0 or alpha > 1.0:
            return {"error": "alpha debe estar entre 0.0 y 1.0."}

        if filtros is not None and not isinstance(filtros, dict):
            return {"error": "filtros debe ser un diccionario de metadatos."}

        try:
            vector = await self.codificador.codificar_texto(texto_pregunta)
            resultados = await self.repo_documentos.consulta_hibrida_sql(
                vector_consulta=vector,
                filtros=filtros or {},
                top_k=top_k,
            )
            return {
                "texto_pregunta": texto_pregunta.strip(),
                "filtros": filtros or {},
                "top_k": top_k,
                "alpha": alpha,
                "resultados": resultados,
            }
        except Exception as exc:
            return {"error": f"Error al ejecutar consulta híbrida: {exc}"}
