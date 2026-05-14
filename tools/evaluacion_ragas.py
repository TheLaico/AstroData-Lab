"""
Módulo de herramientas MCP para evaluación RAGAS en AstroData Lab.

Expone herramientas MCP para evaluar la calidad del sistema RAG:
1. Evaluar respuestas generadas contra consultas originales y contexto recuperado
2. Calcular métricas RAGAS: faithfulness, answer_relevancy, context_recall
3. Obtener historial de evaluaciones por usuario

Implementa el patrón de inyección de dependencias: repositorios se inyectan
para permitir testing con mocks. Sigue SRP: solo lógica de evaluación RAGAS,
no implementa RAG ni transformación de datos (delegada a repositorios).

Las métricas RAGAS permiten medir:
- Fidelidad: precisión factual de la respuesta
- Relevancia: qué tan bien responde a la pregunta
- Recuperación: cobertura de información necesaria en contexto
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, date
from mcp.types import Tool, TextContent
import re
from collections import Counter

from database.repositorio_evaluaciones import RepositorioEvaluaciones
from database.repositorio_consultas import RepositorioConsultas
from models.evaluacion_ragas_entrada_model import EvaluacionRAGASEntrada
from models.evaluacion_ragas_model import EvaluacionRAGAS
from models.resumen_evaluacion_model import ResumenEvaluacion


class ToolsEvaluacionRAGAS:
    """
    Conjunto de herramientas MCP para evaluación de calidad del sistema RAG.
    
    Orquesta el cálculo de métricas RAGAS (Retrieval-Augmented Generation Assessment)
    para evaluar la calidad de respuestas generadas por el sistema RAG.
    
    Implementa lógica para calcular:
    1. Faithfulness: precisión factual respaldada por contexto
    2. Answer Relevancy: relevancia de respuesta para la pregunta
    3. Context Recall: cobertura efectiva de información necesaria
    
    Sigue Responsabilidad Única: esta clase solo calcula métricas RAGAS
    y orquesta persistencia, no implementa lógica RAG ni de dominio.
    
    Atributos:
        _repo_evaluaciones: Repositorio para persistir evaluaciones
        _repo_consultas: Repositorio para obtener consultas originales
    """
    
    def __init__(self) -> None:
        """
        Inicializa las herramientas de evaluación con sus dependencias.
        
        Instancia repositorios para acceso a datos de evaluaciones y consultas.
        """
        self._repo_evaluaciones = RepositorioEvaluaciones()
        self._repo_consultas = RepositorioConsultas()
    
    
    def _normalizar_texto(self, texto: str) -> List[str]:
        """
        Normaliza texto convirtiéndolo a palabras clave lowercase.
        
        Args:
            texto: Texto a normalizar
            
        Returns:
            Lista de palabras clave (lowercased, sin puntuación)
        """
        if not texto:
            return []
        
        # Remover puntuación y convertir a lowercase
        texto_limpio = re.sub(r'[^\w\s]', '', texto.lower())
        # Remover stopwords comunes en español
        stopwords = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se',
            'no', 'por', 'con', 'su', 'para', 'una', 'o', 'del', 'al',
            'los', 'las', 'un', 'una', 'como', 'más', 'o', 'porque',
            'muy', 'sin', 'sobre', 'ser', 'tiene', 'tienen', 'sido',
            'es', 'son', 'estoy', 'estamos', 'están', 'estaré'
        }
        palabras = [p for p in texto_limpio.split() if p and p not in stopwords]
        return palabras
    
    
    def _calcular_faithfulness(
        self,
        respuesta: str,
        contexto_recuperado: List[str]
    ) -> float:
        """
        Calcula fidelidad: proporción de afirmaciones de respuesta respaldadas por contexto.
        
        Estrategia: dividir respuesta en oraciones, medir solapamiento de vocabulario
        con contexto recuperado. Mayor solapamiento → mayor fidelidad.
        
        Args:
            respuesta: Texto de respuesta generada
            contexto_recuperado: Lista de chunks de contexto
            
        Returns:
            Puntuación de fidelidad (0.0-1.0)
        """
        if not respuesta or not contexto_recuperado:
            return 0.5  # Neutral si faltan datos
        
        # Dividir respuesta en oraciones
        oraciones = re.split(r'[.!?]+', respuesta)
        oraciones = [o.strip() for o in oraciones if o.strip()]
        
        if not oraciones:
            return 0.5
        
        # Palabras clave del contexto
        contexto_palabras = set()
        for chunk in contexto_recuperado:
            contexto_palabras.update(self._normalizar_texto(chunk))
        
        if not contexto_palabras:
            return 0.5
        
        # Contar oraciones respaldadas por contexto
        oraciones_respaldadas = 0
        for oracion in oraciones:
            palabras_oracion = set(self._normalizar_texto(oracion))
            # Si al menos 30% de palabras coinciden, se considera respaldada
            if palabras_oracion and len(palabras_oracion & contexto_palabras) / len(palabras_oracion) >= 0.3:
                oraciones_respaldadas += 1
        
        # Retornar proporción
        fidelidad = oraciones_respaldadas / len(oraciones) if oraciones else 0.5
        return min(max(fidelidad, 0.0), 1.0)
    
    
    def _calcular_answer_relevancy(self, pregunta: str, respuesta: str) -> float:
        """
        Calcula relevancia: qué tan bien la respuesta toma en cuenta la pregunta.
        
        Estrategia: medir solapamiento de palabras clave entre pregunta y respuesta.
        
        Args:
            pregunta: Texto de pregunta original
            respuesta: Texto de respuesta
            
        Returns:
            Puntuación de relevancia (0.0-1.0)
        """
        if not pregunta or not respuesta:
            return 0.5
        
        palabras_pregunta = set(self._normalizar_texto(pregunta))
        palabras_respuesta = set(self._normalizar_texto(respuesta))
        
        if not palabras_pregunta:
            return 0.5
        
        # Proporción de palabras clave de pregunta en respuesta
        coincidencias = len(palabras_pregunta & palabras_respuesta)
        relevancia = coincidencias / len(palabras_pregunta)
        
        return min(max(relevancia, 0.0), 1.0)
    
    
    def _calcular_context_recall(
        self,
        respuesta: str,
        contexto_recuperado: List[str]
    ) -> float:
        """
        Calcula context recall: proporción del contexto efectivamente usado.
        
        Estrategia: medir qué porcentaje de palabras clave del contexto
        aparecen en la respuesta.
        
        Args:
            respuesta: Texto de respuesta
            contexto_recuperado: Lista de chunks de contexto
            
        Returns:
            Puntuación de recuperación (0.0-1.0)
        """
        if not contexto_recuperado or not respuesta:
            return 0.5
        
        # Palabras clave del contexto
        palabras_contexto = []
        for chunk in contexto_recuperado:
            palabras_contexto.extend(self._normalizar_texto(chunk))
        
        if not palabras_contexto:
            return 0.5
        
        # Palabras únicas del contexto
        palabras_unicas_contexto = set(palabras_contexto)
        
        # Palabras de respuesta
        palabras_respuesta = set(self._normalizar_texto(respuesta))
        
        # Proporción de palabras del contexto que aparecen en respuesta
        if palabras_unicas_contexto:
            recall = len(palabras_unicas_contexto & palabras_respuesta) / len(palabras_unicas_contexto)
        else:
            recall = 0.5
        
        return min(max(recall, 0.0), 1.0)
    
    
    async def evaluar_respuesta_rag(
        self,
        id_consulta: int,
        respuesta_generada: str,
        contexto_recuperado: List[str],
        modelo_eval: str = 'claude-sonnet-4-20250514'
    ) -> Dict[str, Any]:
        """
        Evalúa una respuesta RAG calculando métricas RAGAS.
        
        Pipeline:
        1. Obtiene consulta original desde BD usando id_consulta
        2. Calcula faithfulness: precisión factual respaldada por contexto
        3. Calcula answer_relevancy: relevancia de respuesta para pregunta
        4. Calcula context_recall: cobertura de información del contexto
        5. Persiste evaluación en BD con registrar_evaluacion_ragas()
        6. Retorna métricas, promedios y clasificación de calidad
        
        Args:
            id_consulta: ID de la consulta a evaluar
            respuesta_generada: Texto de respuesta generada por el sistema
            contexto_recuperado: Lista de chunks de contexto recuperados
            modelo_eval: Nombre del modelo evaluador (default: claude-sonnet)
        
        Returns:
            Dict con estructura:
            {
                'id_evaluacion': int,
                'faithfulness': float (0.0-1.0),
                'answer_relevancy': float (0.0-1.0),
                'context_recall': float (0.0-1.0),
                'promedio_metricas': float (0.0-1.0),
                'calidad': str ('baja', 'media', 'alta'),
                'modelo_eval': str,
                'fecha_evaluacion': str (ISO format)
            }
            
            O en caso de error:
            {
                'error': str,
                'detalles': str
            }
        
        Example:
            >>> resultado = await evaluacion.evaluar_respuesta_rag(
            ...     id_consulta=42,
            ...     respuesta_generada="Los agujeros negros se forman...",
            ...     contexto_recuperado=["Un agujero negro es...", "Se forman cuando..."],
            ...     modelo_eval='claude-sonnet'
            ... )
            >>> resultado['calidad']
            'alta'
        """
        try:
            # 1. Validar inputs básicos
            if id_consulta <= 0:
                return {
                    'error': "id_consulta debe ser positivo",
                    'detalles': f"Valor recibido: {id_consulta}"
                }
            
            if not respuesta_generada or not respuesta_generada.strip():
                return {
                    'error': "respuesta_generada no puede estar vacía",
                    'detalles': ""
                }
            
            if not contexto_recuperado or len(contexto_recuperado) == 0:
                return {
                    'error': "contexto_recuperado no puede estar vacío",
                    'detalles': ""
                }
            
            # 2. Obtener consulta original
            try:
                consulta = await self._repo_consultas.obtener_consulta_por_id(id_consulta)
                if not consulta:
                    return {
                        'error': "Consulta no encontrada",
                        'detalles': f"ID: {id_consulta}"
                    }
            except Exception as e:
                return {
                    'error': "Error al obtener consulta original",
                    'detalles': str(e)
                }
            
            # 3. Calcular métricas RAGAS
            try:
                faithfulness = self._calcular_faithfulness(
                    respuesta_generada.strip(),
                    contexto_recuperado
                )
                
                answer_relevancy = self._calcular_answer_relevancy(
                    consulta.texto_pregunta,
                    respuesta_generada.strip()
                )
                
                context_recall = self._calcular_context_recall(
                    respuesta_generada.strip(),
                    contexto_recuperado
                )
            except Exception as e:
                return {
                    'error': "Error al calcular métricas RAGAS",
                    'detalles': str(e)
                }
            
            # 4. Crear y persistir evaluación
            try:
                entrada = EvaluacionRAGASEntrada(
                    faithfulness=faithfulness,
                    answer_relevancy=answer_relevancy,
                    context_recall=context_recall,
                    modelo_eval=modelo_eval,
                    id_consulta=id_consulta
                )
                
                evaluacion = await self._repo_evaluaciones.registrar_evaluacion_ragas(entrada)
            except Exception as e:
                return {
                    'error': "Error al persistir evaluación",
                    'detalles': str(e)
                }
            
            # 5. Calcular resumen
            promedio = (faithfulness + answer_relevancy + context_recall) / 3.0
            if promedio < 0.4:
                calidad = 'baja'
            elif promedio <= 0.7:
                calidad = 'media'
            else:
                calidad = 'alta'
            
            return {
                'id_evaluacion': evaluacion.id_evaluacion,
                'faithfulness': round(faithfulness, 4),
                'answer_relevancy': round(answer_relevancy, 4),
                'context_recall': round(context_recall, 4),
                'promedio_metricas': round(promedio, 4),
                'calidad': calidad,
                'modelo_eval': modelo_eval,
                'fecha_evaluacion': evaluacion.fecha.isoformat()
            }
        
        except Exception as e:
            return {
                'error': "Error al evaluar respuesta RAG",
                'detalles': str(e)
            }
    
    
    async def obtener_historial_evaluaciones(
        self,
        id_usuario: int,
        fecha_desde: Optional[str] = None,
        fecha_hasta: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtiene historial de evaluaciones de un usuario con resumen agregado.
        
        Pipeline:
        1. Convierte strings de fecha a objetos date si se proveen
        2. Valida que id_usuario sea válido
        3. Llama a listar_evaluaciones_por_usuario() con rango de fechas
        4. Calcula ResumenEvaluacion del usuario (promedios y calidad)
        5. Retorna lista de evaluaciones + resumen agregado
        
        Útil para dashboards y reportes de desempeño del sistema RAG.
        
        Args:
            id_usuario: ID del usuario
            fecha_desde: Fecha inicial en formato ISO (ej: "2026-01-01")
                        Si None, sin límite inferior
            fecha_hasta: Fecha final en formato ISO (ej: "2026-05-31")
                        Si None, sin límite superior
        
        Returns:
            Dict con estructura:
            {
                'id_usuario': int,
                'evaluaciones': [
                    {
                        'id_evaluacion': int,
                        'faithfulness': float,
                        'answer_relevancy': float,
                        'context_recall': float,
                        'promedio': float,
                        'calidad': str,
                        'modelo_eval': str,
                        'fecha': str (ISO format)
                    },
                    ...
                ],
                'total_evaluaciones': int,
                'resumen_usuario': {
                    'faithfulness_promedio': float,
                    'answer_relevancy_promedio': float,
                    'context_recall_promedio': float,
                    'promedio_general': float,
                    'calidad_general': str
                },
                'rango_fechas': {
                    'desde': str | None,
                    'hasta': str | None
                }
            }
            
            O en caso de error:
            {
                'error': str,
                'detalles': str,
                'evaluaciones': [],
                'total_evaluaciones': 0
            }
        
        Example:
            >>> resultado = await evaluacion.obtener_historial_evaluaciones(
            ...     id_usuario=1,
            ...     fecha_desde="2026-01-01",
            ...     fecha_hasta="2026-05-31"
            ... )
            >>> resultado['resumen_usuario']['calidad_general']
            'alta'
        """
        try:
            # 1. Validar id_usuario
            if id_usuario <= 0:
                return {
                    'error': "id_usuario debe ser positivo",
                    'detalles': f"Valor recibido: {id_usuario}",
                    'evaluaciones': [],
                    'total_evaluaciones': 0
                }
            
            # 2. Convertir fechas
            fecha_desde_date = None
            fecha_hasta_date = None
            
            try:
                if fecha_desde:
                    fecha_desde_date = datetime.fromisoformat(fecha_desde).date()
            except (ValueError, TypeError) as e:
                return {
                    'error': "Formato de fecha_desde inválido",
                    'detalles': f"Debe ser ISO format (ej: 2026-01-01), recibido: {fecha_desde}",
                    'evaluaciones': [],
                    'total_evaluaciones': 0
                }
            
            try:
                if fecha_hasta:
                    fecha_hasta_date = datetime.fromisoformat(fecha_hasta).date()
            except (ValueError, TypeError) as e:
                return {
                    'error': "Formato de fecha_hasta inválido",
                    'detalles': f"Debe ser ISO format (ej: 2026-12-31), recibido: {fecha_hasta}",
                    'evaluaciones': [],
                    'total_evaluaciones': 0
                }
            
            # 3. Obtener evaluaciones
            try:
                evaluaciones = await self._repo_evaluaciones.listar_evaluaciones_por_usuario(
                    id_usuario=id_usuario,
                    fecha_desde=fecha_desde_date,
                    fecha_hasta=fecha_hasta_date
                )
            except Exception as e:
                return {
                    'error': "Error al listar evaluaciones del usuario",
                    'detalles': str(e),
                    'evaluaciones': [],
                    'total_evaluaciones': 0
                }
            
            # 4. Formatear evaluaciones
            evaluaciones_lista = []
            for eval in evaluaciones:
                promedio = (eval.faithfulness + eval.answer_relevancy + eval.context_recall) / 3.0
                
                if promedio < 0.4:
                    calidad = 'baja'
                elif promedio <= 0.7:
                    calidad = 'media'
                else:
                    calidad = 'alta'
                
                evaluaciones_lista.append({
                    'id_evaluacion': eval.id_evaluacion,
                    'faithfulness': round(eval.faithfulness, 4),
                    'answer_relevancy': round(eval.answer_relevancy, 4),
                    'context_recall': round(eval.context_recall, 4),
                    'promedio': round(promedio, 4),
                    'calidad': calidad,
                    'modelo_eval': eval.modelo_eval,
                    'fecha': eval.fecha.isoformat()
                })
            
            # 5. Calcular resumen del usuario
            resumen_usuario = None
            try:
                resumen = await self._repo_evaluaciones.calcular_resumen_usuario(id_usuario)
                
                resumen_usuario = {
                    'faithfulness_promedio': round(resumen.evaluacion.faithfulness, 4),
                    'answer_relevancy_promedio': round(resumen.evaluacion.answer_relevancy, 4),
                    'context_recall_promedio': round(resumen.evaluacion.context_recall, 4),
                    'promedio_general': round(resumen.promedio_metricas, 4),
                    'calidad_general': resumen.calidad
                }
            except Exception as e:
                # No es fatal si falla el resumen
                resumen_usuario = None
            
            return {
                'id_usuario': id_usuario,
                'evaluaciones': evaluaciones_lista,
                'total_evaluaciones': len(evaluaciones_lista),
                'resumen_usuario': resumen_usuario,
                'rango_fechas': {
                    'desde': fecha_desde,
                    'hasta': fecha_hasta
                }
            }
        
        except Exception as e:
            return {
                'error': "Error al obtener historial de evaluaciones",
                'detalles': str(e),
                'evaluaciones': [],
                'total_evaluaciones': 0
            }
    
    
    def obtener_definiciones_tools(self) -> List[Tool]:
        """
        Retorna las definiciones de herramientas MCP para registro en el servidor.
        
        Genera Tool objects que describen a Claude la interfaz de cada herramienta
        de evaluación, incluyendo nombres, descripciones, y esquemas de entrada JSON.
        
        Returns:
            List[Tool] con 2 definiciones de herramientas MCP
        
        Estructura de cada Tool:
            - name: Identificador de herramienta para Claude
            - description: Descripción en español de qué hace
            - inputSchema: JSON Schema describiendo los parámetros
        """
        return [
            Tool(
                name="evaluar_respuesta_rag",
                description="Evalúa una respuesta RAG calculando métricas RAGAS: fidelidad, relevancia y recuperación.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_consulta": {
                            "type": "integer",
                            "description": "ID de la consulta evaluada"
                        },
                        "respuesta_generada": {
                            "type": "string",
                            "description": "Texto de respuesta generada por el sistema"
                        },
                        "contexto_recuperado": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Lista de chunks de contexto recuperados"
                        },
                        "modelo_eval": {
                            "type": "string",
                            "description": "Nombre del modelo evaluador (default: claude-sonnet-4-20250514)"
                        }
                    },
                    "required": ["id_consulta", "respuesta_generada", "contexto_recuperado"]
                }
            ),
            Tool(
                name="obtener_historial_evaluaciones",
                description="Obtiene el historial de evaluaciones RAGAS de un usuario con resumen agregado.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_usuario": {
                            "type": "integer",
                            "description": "ID del usuario"
                        },
                        "fecha_desde": {
                            "type": "string",
                            "description": "Fecha inicial en formato ISO (ej: '2026-01-01', opcional)"
                        },
                        "fecha_hasta": {
                            "type": "string",
                            "description": "Fecha final en formato ISO (ej: '2026-12-31', opcional)"
                        }
                    },
                    "required": ["id_usuario"]
                }
            )
        ]
