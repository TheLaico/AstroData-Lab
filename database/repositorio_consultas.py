"""
Módulo de repositorio para consultas y resultados en AstroData Lab.

Proporciona la capa de acceso a datos para el pipeline RAG: gestiona consultas de usuarios,
embeddings de consultas, resultados recuperados y su evaluación. Centraliza toda la lógica
de persistencia relacionada con el flujo de pregunta-respuesta del sistema.
"""

from typing import List, Optional
from datetime import datetime
from database.conexion import conexion_bd
from models.consulta import ConsultaEntrada, Consulta, Usuario
from models.resultado import Resultado, ResultadoDetallado, Documento, Imagen


class RepositorioConsultas:
    """
    Repositorio para gestionar consultas, resultados y embeddings en el sistema RAG.
    
    Encapsula operaciones CRUD sobre la tabla Consulta, Embedding_Consulta y Resultado.
    Diseñado con OCP para permitir extensión sin modificación.
    
    El flujo típico es:
    1. Registrar consulta del usuario → registrar_consulta()
    2. Generar embedding de la consulta → guardar_embedding_consulta()
    3. Buscar resultados (via RepositorioDocumentos) → registrar_resultado() para cada uno
    4. Recuperar resultados completos → obtener_resultados_por_consulta()
    """
    
    # ========================================================================
    # OPERACIONES CRUD DE CONSULTA
    # ========================================================================
    
    async def registrar_consulta(self, datos: ConsultaEntrada) -> Consulta:
        """
        Registra una nueva consulta del usuario en la base de datos.
        
        Inserta en tabla Consulta con texto y usuario. La BD asigna automáticamente
        id_consulta y fecha actual. Se utiliza al inicio del pipeline RAG cuando
        el usuario realiza una pregunta.
        
        SQL utilizado:
            INSERT INTO Consulta (texto_pregunta, id_usuario)
            VALUES ($1, $2)
            RETURNING id_consulta, texto_pregunta, fecha, id_usuario
        
        Args:
            datos: ConsultaEntrada con texto_pregunta e id_usuario
        
        Returns:
            Consulta completa con id_consulta y fecha asignados por la BD
        
        Raises:
            ValueError: Si el texto o usuario son inválidos
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioConsultas()
            >>> entrada = ConsultaEntrada(
            ...     texto_pregunta="¿Qué planetas son habitables?",
            ...     id_usuario=1
            ... )
            >>> consulta = await repo.registrar_consulta(entrada)
            >>> consulta.id_consulta  # Asignado por BD
            42
        """
        if not datos.texto_pregunta or not datos.texto_pregunta.strip():
            raise ValueError("El texto de la pregunta no puede estar vacío")
        
        if datos.id_usuario <= 0:
            raise ValueError("id_usuario debe ser positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                fila = await conexion.fetchrow(
                    """
                    INSERT INTO Consulta (texto_pregunta, id_usuario)
                    VALUES ($1, $2)
                    RETURNING id_consulta, texto_pregunta, fecha, id_usuario
                    """,
                    datos.texto_pregunta.strip(),
                    datos.id_usuario
                )
                
                return Consulta(
                    id_consulta=fila['id_consulta'],
                    texto_pregunta=fila['texto_pregunta'],
                    fecha=fila['fecha'],
                    id_usuario=fila['id_usuario']
                )
        
        except Exception as e:
            raise RuntimeError(
                f"Error al registrar consulta: {e}"
            ) from e
    
    async def obtener_consulta_por_id(self, id_consulta: int) -> Optional[Consulta]:
        """
        Recupera una consulta por su identificador único.
        
        SQL utilizado:
            SELECT id_consulta, texto_pregunta, fecha, id_usuario
            FROM Consulta
            WHERE id_consulta = $1
        
        Args:
            id_consulta: ID de la consulta a buscar
        
        Returns:
            Consulta si existe, None en caso contrario
        
        Raises:
            ValueError: Si id_consulta no es positivo
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioConsultas()
            >>> consulta = await repo.obtener_consulta_por_id(42)
            >>> consulta.texto_pregunta if consulta else "No encontrada"
        """
        if not isinstance(id_consulta, int) or id_consulta <= 0:
            raise ValueError("id_consulta debe ser un entero positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                fila = await conexion.fetchrow(
                    """
                    SELECT id_consulta, texto_pregunta, fecha, id_usuario
                    FROM Consulta
                    WHERE id_consulta = $1
                    """,
                    id_consulta
                )
                
                if not fila:
                    return None
                
                return Consulta(
                    id_consulta=fila['id_consulta'],
                    texto_pregunta=fila['texto_pregunta'],
                    fecha=fila['fecha'],
                    id_usuario=fila['id_usuario']
                )
        
        except Exception as e:
            raise RuntimeError(
                f"Error al obtener consulta {id_consulta}: {e}"
            ) from e
    
    async def listar_consultas_por_usuario(
        self,
        id_usuario: int,
        limite: int = 20
    ) -> List[Consulta]:
        """
        Lista todas las consultas realizadas por un usuario específico.
        
        Retorna consultas ordenadas por fecha descendente (más recientes primero).
        Útil para historial de usuario y debugging.
        
        SQL utilizado:
            SELECT id_consulta, texto_pregunta, fecha, id_usuario
            FROM Consulta
            WHERE id_usuario = $1
            ORDER BY fecha DESC
            LIMIT $2
        
        Args:
            id_usuario: ID del usuario
            limite: Número máximo de consultas a retornar (default: 20)
        
        Returns:
            Lista de Consulta ordenadas por fecha descendente
        
        Raises:
            ValueError: Si los parámetros no son válidos
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioConsultas()
            >>> consultas = await repo.listar_consultas_por_usuario(1, limite=10)
            >>> len(consultas)
            10
            >>> consultas[0].fecha > consultas[1].fecha  # Más recientes primero
            True
        """
        if id_usuario <= 0:
            raise ValueError("id_usuario debe ser positivo")
        
        if limite <= 0:
            raise ValueError("limite debe ser positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                filas = await conexion.fetch(
                    """
                    SELECT id_consulta, texto_pregunta, fecha, id_usuario
                    FROM Consulta
                    WHERE id_usuario = $1
                    ORDER BY fecha DESC
                    LIMIT $2
                    """,
                    id_usuario,
                    limite
                )
                
                consultas = []
                for fila in filas:
                    consultas.append(
                        Consulta(
                            id_consulta=fila['id_consulta'],
                            texto_pregunta=fila['texto_pregunta'],
                            fecha=fila['fecha'],
                            id_usuario=fila['id_usuario']
                        )
                    )
                
                return consultas
        
        except Exception as e:
            raise RuntimeError(
                f"Error al listar consultas del usuario {id_usuario}: {e}"
            ) from e
    
    # ========================================================================
    # OPERACIONES CRUD DE RESULTADO
    # ========================================================================
    
    async def registrar_resultado(self, datos: Resultado) -> Resultado:
        """
        Registra un resultado de búsqueda RAG para una consulta.
        
        Inserta en tabla Resultado con descripción, relevancia y referencias a
        documento/imagen recuperados. Se llama para cada resultado encontrado.
        
        SQL utilizado:
            INSERT INTO Resultado (descripcion_resultado, relevancia, id_consulta, id_doc, id_imagen)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id_resultado
        
        Args:
            datos: Resultado con descripción, relevancia, id_consulta y referencias opcionales
        
        Returns:
            Resultado completo con id_resultado asignado
        
        Raises:
            ValueError: Si los parámetros no son válidos
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioConsultas()
            >>> resultado = Resultado(
            ...     id_resultado=999,  # Ignorado
            ...     descripcion_resultado="Tierra es un planeta habitable",
            ...     relevancia=0.95,
            ...     id_consulta=42,
            ...     id_doc=1,
            ...     id_imagen=None
            ... )
            >>> nuevo = await repo.registrar_resultado(resultado)
        """
        if datos.id_consulta <= 0:
            raise ValueError("id_consulta debe ser positivo")
        
        if not (0.0 <= datos.relevancia <= 1.0):
            raise ValueError("relevancia debe estar entre 0.0 y 1.0")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_resultado = await conexion.fetchval(
                    """
                    INSERT INTO Resultado (descripcion_resultado, relevancia, id_consulta, id_doc, id_imagen)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id_resultado
                    """,
                    datos.descripcion_resultado,
                    datos.relevancia,
                    datos.id_consulta,
                    datos.id_doc,
                    datos.id_imagen
                )
                
                datos.id_resultado = id_resultado
                return datos
        
        except Exception as e:
            raise RuntimeError(
                f"Error al registrar resultado: {e}"
            ) from e
    
    async def obtener_resultados_por_consulta(
        self,
        id_consulta: int
    ) -> List[ResultadoDetallado]:
        """
        Recupera todos los resultados de una consulta con datos completos.
        
        Realiza JOINs con Documento e Imagen para retornar ResultadoDetallado
        con objetos completos, permitiendo mostrar información enriquecida al usuario.
        
        SQL utilizado:
            SELECT r.id_resultado, r.descripcion_resultado, r.relevancia,
                   r.id_consulta, r.id_doc, r.id_imagen,
                   d.id_doc, d.titulo, d.idioma, d.fecha, d.fuente,
                   d.contenido_texto, d.id_objeto,
                   i.id_imagen, i.ruta_archivo, i.descripcion, i.etiquetas, i.id_doc
            FROM Resultado r
            LEFT JOIN Documento d ON r.id_doc = d.id_doc
            LEFT JOIN Imagen i ON r.id_imagen = i.id_imagen
            WHERE r.id_consulta = $1
            ORDER BY r.relevancia DESC
        
        Args:
            id_consulta: ID de la consulta cuyos resultados se recuperan
        
        Returns:
            Lista de ResultadoDetallado ordenados por relevancia descendente
        
        Raises:
            ValueError: Si id_consulta no es válido
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioConsultas()
            >>> resultados = await repo.obtener_resultados_por_consulta(42)
            >>> len(resultados)
            5
            >>> resultados[0].documento.titulo if resultados[0].documento else None
        """
        if id_consulta <= 0:
            raise ValueError("id_consulta debe ser positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                filas = await conexion.fetch(
                    """
                    SELECT r.id_resultado, r.descripcion_resultado, r.relevancia,
                           r.id_consulta, r.id_doc, r.id_imagen,
                           d.id_doc as d_id_doc, d.titulo, d.idioma, d.fecha, d.fuente,
                           d.contenido_texto, d.id_objeto,
                           i.id_imagen as i_id_imagen, i.ruta_archivo, i.descripcion as i_descripcion,
                           i.etiquetas, i.id_doc as i_id_doc
                    FROM Resultado r
                    LEFT JOIN Documento d ON r.id_doc = d.id_doc
                    LEFT JOIN Imagen i ON r.id_imagen = i.id_imagen
                    WHERE r.id_consulta = $1
                    ORDER BY r.relevancia DESC
                    """,
                    id_consulta
                )
                
                resultados_detallados = []
                for fila in filas:
                    # Construir Documento si existe
                    documento = None
                    if fila['d_id_doc']:
                        documento = Documento(
                            id_doc=fila['d_id_doc'],
                            titulo=fila['titulo'],
                            idioma=fila['idioma'],
                            fecha=fila['fecha'],
                            fuente=fila['fuente'],
                            contenido_texto=fila['contenido_texto'],
                            id_objeto=fila['id_objeto']
                        )
                    
                    # Construir Imagen si existe
                    imagen = None
                    if fila['i_id_imagen']:
                        imagen = Imagen(
                            id_imagen=fila['i_id_imagen'],
                            ruta_archivo=fila['ruta_archivo'],
                            descripcion=fila['i_descripcion'],
                            etiquetas=fila['etiquetas'],
                            id_doc=fila['i_id_doc']
                        )
                    
                    # Construir ResultadoDetallado
                    resultado_base = Resultado(
                        id_resultado=fila['id_resultado'],
                        descripcion_resultado=fila['descripcion_resultado'],
                        relevancia=fila['relevancia'],
                        id_consulta=fila['id_consulta'],
                        id_doc=fila['id_doc'],
                        id_imagen=fila['id_imagen']
                    )
                    
                    resultado_detallado = ResultadoDetallado(
                        **resultado_base.model_dump(),
                        documento=documento,
                        imagen=imagen
                    )
                    
                    resultados_detallados.append(resultado_detallado)
                
                return resultados_detallados
        
        except Exception as e:
            raise RuntimeError(
                f"Error al obtener resultados de consulta {id_consulta}: {e}"
            ) from e