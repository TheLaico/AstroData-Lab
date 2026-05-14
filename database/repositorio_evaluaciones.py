"""
Módulo de repositorio para evaluaciones en AstroData Lab.

Proporciona la capa de acceso a datos para evaluaciones del sistema RAG (métricas RAGAS)
y evaluaciones de habitabilidad planetaria. Gestiona registros, consultas y análisis
agregados de calidad del sistema.
"""

from typing import List, Optional
from datetime import date, datetime
from database.conexion import conexion_bd
from models.evaluacion_ragas_model import EvaluacionRAGAS
from models.evaluacion_ragas_entrada_model import EvaluacionRAGASEntrada
from models.evaluacion_habitabilidad_model import EvaluacionHabitabilidad
from models.resumen_evaluacion_model import ResumenEvaluacion


class RepositorioEvaluaciones:
    """
    Repositorio para gestionar evaluaciones RAGAS y de habitabilidad.
    
    Encapsula operaciones CRUD sobre tablas Evaluacion (métricas RAGAS) y
    Evaluacion_Habitabilidad. Proporciona análisis agregados para monitorear
    calidad del sistema RAG y potencial de habitabilidad planetaria.
    
    Diseñado con OCP para permitir extensión sin modificación.
    
    Las evaluaciones RAGAS permiten:
    - Medir fidelidad, relevancia y recuperación de cada consulta
    - Comparar modelos de evaluación diferentes
    - Rastrear evolución de calidad en el tiempo
    
    Las evaluaciones de habitabilidad permiten:
    - Registrar análisis de potencial habitable de planetas
    - Construir historial de evaluaciones por planeta
    - Comparar condiciones entre planetas similares
    """
    
    # OPERACIONES CRUD DE EVALUACIÓN RAGAS
    
    async def registrar_evaluacion_ragas(
        self,
        datos: EvaluacionRAGASEntrada
    ) -> EvaluacionRAGAS:
        """
        Registra una nueva evaluación RAGAS para una consulta.
        
        Inserta en tabla Evaluacion las tres métricas (faithfulness, answer_relevancy,
        context_recall) junto con el modelo evaluador. La BD asigna automáticamente
        id_evaluacion y fecha actual.
        
        SQL utilizado:
            INSERT INTO Evaluacion (faithfulness, answer_relevancy, context_recall,
                                   modelo_eval, id_consulta)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id_evaluacion, faithfulness, answer_relevancy, context_recall,
                     modelo_eval, fecha, id_consulta
        
        Args:
            datos: EvaluacionRAGASEntrada con las tres métricas, modelo e id_consulta
        
        Returns:
            EvaluacionRAGAS completa con id_evaluacion y fecha asignados
        
        Raises:
            ValueError: Si los parámetros no son válidos
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioEvaluaciones()
            >>> entrada = EvaluacionRAGASEntrada(
            ...     faithfulness=0.95,
            ...     answer_relevancy=0.87,
            ...     context_recall=0.92,
            ...     modelo_eval='gpt-3.5-turbo',
            ...     id_consulta=42
            ... )
            >>> evaluacion = await repo.registrar_evaluacion_ragas(entrada)
        """
        if datos.id_consulta <= 0:
            raise ValueError("id_consulta debe ser positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                fila = await conexion.fetchrow(
                    """
                    INSERT INTO Evaluacion (faithfulness, answer_relevancy, context_recall,
                                           modelo_eval, id_consulta)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id_evaluacion, faithfulness, answer_relevancy, context_recall,
                             modelo_eval, fecha, id_consulta
                    """,
                    datos.faithfulness,
                    datos.answer_relevancy,
                    datos.context_recall,
                    datos.modelo_eval,
                    datos.id_consulta
                )
                
                return EvaluacionRAGAS(
                    id_evaluacion=fila['id_evaluacion'],
                    faithfulness=fila['faithfulness'],
                    answer_relevancy=fila['answer_relevancy'],
                    context_recall=fila['context_recall'],
                    modelo_eval=fila['modelo_eval'],
                    fecha=fila['fecha'],
                    id_consulta=fila['id_consulta']
                )
        
        except Exception as e:
            raise RuntimeError(
                f"Error al registrar evaluación RAGAS: {e}"
            ) from e
    
    async def obtener_evaluacion_por_consulta(
        self,
        id_consulta: int
    ) -> Optional[EvaluacionRAGAS]:
        """
        Recupera la evaluación RAGAS de una consulta específica.
        
        SQL utilizado:
            SELECT id_evaluacion, faithfulness, answer_relevancy, context_recall,
                   modelo_eval, fecha, id_consulta
            FROM Evaluacion
            WHERE id_consulta = $1
            LIMIT 1
        
        Args:
            id_consulta: ID de la consulta cuya evaluación se busca
        
        Returns:
            EvaluacionRAGAS si existe, None en caso contrario
        
        Raises:
            ValueError: Si id_consulta no es válido
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioEvaluaciones()
            >>> evaluacion = await repo.obtener_evaluacion_por_consulta(42)
            >>> evaluacion.faithfulness if evaluacion else None
        """
        if id_consulta <= 0:
            raise ValueError("id_consulta debe ser positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                fila = await conexion.fetchrow(
                    """
                    SELECT id_evaluacion, faithfulness, answer_relevancy, context_recall,
                           modelo_eval, fecha, id_consulta
                    FROM Evaluacion
                    WHERE id_consulta = $1
                    LIMIT 1
                    """,
                    id_consulta
                )
                
                if not fila:
                    return None
                
                return EvaluacionRAGAS(
                    id_evaluacion=fila['id_evaluacion'],
                    faithfulness=fila['faithfulness'],
                    answer_relevancy=fila['answer_relevancy'],
                    context_recall=fila['context_recall'],
                    modelo_eval=fila['modelo_eval'],
                    fecha=fila['fecha'],
                    id_consulta=fila['id_consulta']
                )
        
        except Exception as e:
            raise RuntimeError(
                f"Error al obtener evaluación de consulta {id_consulta}: {e}"
            ) from e
    
    async def listar_evaluaciones_por_usuario(
        self,
        id_usuario: int,
        fecha_desde: Optional[date] = None,
        fecha_hasta: Optional[date] = None
    ) -> List[EvaluacionRAGAS]:
        """
        Lista evaluaciones RAGAS de un usuario en un rango de fechas.
        
        Realiza JOIN con Consulta para filtrar por id_usuario y rango de fechas.
        Ordenadas por fecha descendente (más recientes primero).
        
        SQL utilizado (sin filtro de fechas):
            SELECT e.id_evaluacion, e.faithfulness, e.answer_relevancy, e.context_recall,
                   e.modelo_eval, e.fecha, e.id_consulta
            FROM Evaluacion e
            JOIN Consulta c ON e.id_consulta = c.id_consulta
            WHERE c.id_usuario = $1
            ORDER BY e.fecha DESC
        
        SQL utilizado (con filtro de fechas):
            SELECT e.id_evaluacion, e.faithfulness, e.answer_relevancy, e.context_recall,
                   e.modelo_eval, e.fecha, e.id_consulta
            FROM Evaluacion e
            JOIN Consulta c ON e.id_consulta = c.id_consulta
            WHERE c.id_usuario = $1
            AND e.fecha >= $2::date
            AND e.fecha < $3::date + interval '1 day'
            ORDER BY e.fecha DESC
        
        Args:
            id_usuario: ID del usuario
            fecha_desde: Fecha inicial del rango (incluida), opcional
            fecha_hasta: Fecha final del rango (incluida), opcional
        
        Returns:
            Lista de EvaluacionRAGAS ordenadas por fecha descendente
        
        Raises:
            ValueError: Si los parámetros no son válidos
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioEvaluaciones()
            >>> evaluaciones = await repo.listar_evaluaciones_por_usuario(
            ...     id_usuario=1,
            ...     fecha_desde=date(2026, 1, 1),
            ...     fecha_hasta=date(2026, 5, 31)
            ... )
            >>> len(evaluaciones)
            15
        """
        if id_usuario <= 0:
            raise ValueError("id_usuario debe ser positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                if fecha_desde or fecha_hasta:
                    # Con filtro de fechas
                    filas = await conexion.fetch(
                        """
                        SELECT e.id_evaluacion, e.faithfulness, e.answer_relevancy,
                               e.context_recall, e.modelo_eval, e.fecha, e.id_consulta
                        FROM Evaluacion e
                        JOIN Consulta c ON e.id_consulta = c.id_consulta
                        WHERE c.id_usuario = $1
                        AND e.fecha >= $2::date
                        AND e.fecha < $3::date + interval '1 day'
                        ORDER BY e.fecha DESC
                        """,
                        id_usuario,
                        fecha_desde or date.min,
                        fecha_hasta or date.max
                    )
                else:
                    # Sin filtro de fechas
                    filas = await conexion.fetch(
                        """
                        SELECT e.id_evaluacion, e.faithfulness, e.answer_relevancy,
                               e.context_recall, e.modelo_eval, e.fecha, e.id_consulta
                        FROM Evaluacion e
                        JOIN Consulta c ON e.id_consulta = c.id_consulta
                        WHERE c.id_usuario = $1
                        ORDER BY e.fecha DESC
                        """,
                        id_usuario
                    )
                
                evaluaciones = []
                for fila in filas:
                    evaluaciones.append(
                        EvaluacionRAGAS(
                            id_evaluacion=fila['id_evaluacion'],
                            faithfulness=fila['faithfulness'],
                            answer_relevancy=fila['answer_relevancy'],
                            context_recall=fila['context_recall'],
                            modelo_eval=fila['modelo_eval'],
                            fecha=fila['fecha'],
                            id_consulta=fila['id_consulta']
                        )
                    )
                
                return evaluaciones
        
        except Exception as e:
            raise RuntimeError(
                f"Error al listar evaluaciones del usuario {id_usuario}: {e}"
            ) from e
    
    async def calcular_resumen_usuario(self, id_usuario: int) -> ResumenEvaluacion:
        """
        Calcula un resumen agregado de métricas RAGAS para un usuario.
        
        Promedia todas las evaluaciones RAGAS del usuario y proporciona clasificación
        de calidad general (baja/media/alta). Útil para dashboards y reportes.
        
        SQL utilizado:
            SELECT AVG(e.faithfulness) as avg_faithfulness,
                   AVG(e.answer_relevancy) as avg_answer_relevancy,
                   AVG(e.context_recall) as avg_context_recall,
                   'all-MiniLM-L6-v2' as modelo_eval,
                   COUNT(*) as num_evaluaciones
            FROM Evaluacion e
            JOIN Consulta c ON e.id_consulta = c.id_consulta
            WHERE c.id_usuario = $1
        
        Args:
            id_usuario: ID del usuario
        
        Returns:
            ResumenEvaluacion con evaluación promediada y clasificación de calidad
        
        Raises:
            ValueError: Si id_usuario no es válido
            RuntimeError: Si hay error en la operación de BD o sin evaluaciones
        
        Example:
            >>> repo = RepositorioEvaluaciones()
            >>> resumen = await repo.calcular_resumen_usuario(1)
            >>> resumen.promedio_metricas
            0.91
            >>> resumen.calidad
            'alta'
        """
        if id_usuario <= 0:
            raise ValueError("id_usuario debe ser positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                fila = await conexion.fetchrow(
                    """
                    SELECT AVG(e.faithfulness)::float as avg_faithfulness,
                           AVG(e.answer_relevancy)::float as avg_answer_relevancy,
                           AVG(e.context_recall)::float as avg_context_recall,
                           COUNT(*) as num_evaluaciones
                    FROM Evaluacion e
                    JOIN Consulta c ON e.id_consulta = c.id_consulta
                    WHERE c.id_usuario = $1
                    """,
                    id_usuario
                )
                
                if not fila or fila['num_evaluaciones'] == 0:
                    raise ValueError(
                        f"No hay evaluaciones RAGAS para el usuario {id_usuario}"
                    )
                
                # Crear EvaluacionRAGAS ficticia con promedios
                evaluacion_promedio = EvaluacionRAGAS(
                    id_evaluacion=-1,  # Ficticia
                    faithfulness=float(fila['avg_faithfulness']),
                    answer_relevancy=float(fila['avg_answer_relevancy']),
                    context_recall=float(fila['avg_context_recall']),
                    modelo_eval=f"promedio_de_{fila['num_evaluaciones']}_evaluaciones",
                    fecha=datetime.now(),
                    id_consulta=-1  # Ficticia
                )
                
                return ResumenEvaluacion(evaluacion=evaluacion_promedio)
        
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Error al calcular resumen del usuario {id_usuario}: {e}"
            ) from e
    
    # OPERACIONES CRUD DE EVALUACIÓN DE HABITABILIDAD
    
    async def registrar_evaluacion_habitabilidad(
        self,
        datos: EvaluacionHabitabilidad
    ) -> EvaluacionHabitabilidad:
        """
        Registra una nueva evaluación de habitabilidad planetaria.
        
        Inserta en tabla Evaluacion_Habitabilidad el análisis de potencial habitable
        de un planeta con puntaje, descripción y fecha.
        
        SQL utilizado:
            INSERT INTO Evaluacion_Habitabilidad (id_planeta, puntaje, descripcion, fecha)
            VALUES ($1, $2, $3, $4)
            RETURNING id_eval_habitabilidad, id_planeta, puntaje, descripcion, fecha
        
        Args:
            datos: EvaluacionHabitabilidad con puntaje, descripción e id_planeta
        
        Returns:
            EvaluacionHabitabilidad creada con id_eval_habitabilidad asignado
        
        Raises:
            ValueError: Si los parámetros no son válidos
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioEvaluaciones()
            >>> evaluacion = EvaluacionHabitabilidad(
            ...     id_eval_habitabilidad=999,  # Ignorado
            ...     id_planeta=1,
            ...     puntaje=0.95,
            ...     descripcion="Planeta altamente habitable con agua líquida",
            ...     fecha=datetime.now()
            ... )
            >>> nueva = await repo.registrar_evaluacion_habitabilidad(evaluacion)
        """
        if datos.id_planeta <= 0:
            raise ValueError("id_planeta debe ser positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                fila = await conexion.fetchrow(
                    """
                    INSERT INTO Evaluacion_Habitabilidad (id_planeta, puntaje, descripcion, fecha)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id_eval_habitabilidad, id_planeta, puntaje, descripcion, fecha
                    """,
                    datos.id_planeta,
                    datos.puntaje,
                    datos.descripcion,
                    datos.fecha
                )
                
                return EvaluacionHabitabilidad(
                    id_eval_habitabilidad=fila['id_eval_habitabilidad'],
                    id_planeta=fila['id_planeta'],
                    puntaje=fila['puntaje'],
                    descripcion=fila['descripcion'],
                    fecha=fila['fecha']
                )
        
        except Exception as e:
            raise RuntimeError(
                f"Error al registrar evaluación de habitabilidad: {e}"
            ) from e
    
    async def obtener_historial_habitabilidad(
        self,
        id_planeta: int
    ) -> List[EvaluacionHabitabilidad]:
        """
        Recupera el historial completo de evaluaciones de habitabilidad de un planeta.
        
        Retorna todas las evaluaciones ordenadas por fecha descendente, permitiendo
        ver evolución temporal del análisis de habitabilidad.
        
        SQL utilizado:
            SELECT id_eval_habitabilidad, id_planeta, puntaje, descripcion, fecha
            FROM Evaluacion_Habitabilidad
            WHERE id_planeta = $1
            ORDER BY fecha DESC
        
        Args:
            id_planeta: ID del planeta cuyo historial se recupera
        
        Returns:
            Lista de EvaluacionHabitabilidad ordenadas por fecha descendente
        
        Raises:
            ValueError: Si id_planeta no es válido
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioEvaluaciones()
            >>> historial = await repo.obtener_historial_habitabilidad(1)
            >>> len(historial)
            5
            >>> historial[0].fecha > historial[1].fecha  # Más recientes primero
            True
        """
        if id_planeta <= 0:
            raise ValueError("id_planeta debe ser positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                filas = await conexion.fetch(
                    """
                    SELECT id_eval_habitabilidad, id_planeta, puntaje, descripcion, fecha
                    FROM Evaluacion_Habitabilidad
                    WHERE id_planeta = $1
                    ORDER BY fecha DESC
                    """,
                    id_planeta
                )
                
                evaluaciones = []
                for fila in filas:
                    evaluaciones.append(
                        EvaluacionHabitabilidad(
                            id_eval_habitabilidad=fila['id_eval_habitabilidad'],
                            id_planeta=fila['id_planeta'],
                            puntaje=fila['puntaje'],
                            descripcion=fila['descripcion'],
                            fecha=fila['fecha']
                        )
                    )
                
                return evaluaciones
        
        except Exception as e:
            raise RuntimeError(
                f"Error al obtener historial de habitabilidad del planeta {id_planeta}: {e}"
            ) from e
