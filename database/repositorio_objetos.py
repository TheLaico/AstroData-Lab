"""
Módulo de repositorio para objetos astronómicos en AstroData Lab.

Proporciona la capa de acceso a datos (Data Access Layer) para todas las operaciones
CRUD sobre la jerarquía de objetos astronómicos (galaxias, sistemas, estrellas, planetas, lunas).
Encapsula consultas SQL y manejo de transacciones con asyncpg.
"""

from typing import List, Optional
from database.conexion import conexion_bd
from models.objeto_astronomico import (
    ObjetoAstronomico,
    Galaxia,
    SistemaEstelar,
    Estrella,
    Planeta,
    Luna,
    CaracteristicaAmbiental,
)
from models.evaluacion import EvaluacionHabitabilidad

class RepositorioObjetos:
    """
    Repositorio para gestionar objetos astronómicos en la base de datos.
    
    Encapsula todas las operaciones CRUD sobre la tabla Objeto_Astronomico y sus
    subtipos (Galaxia, Sistema_Estelar, Estrella, Planeta, Luna). Diseñado con
    el Principio Abierto/Cerrado (OCP): puede extenderse con nuevos métodos sin
    modificar los existentes.
    
    Utiliza el patrón Repository para:
    - Centralizar lógica de acceso a datos
    - Permitir testing con mocks
    - Aislar cambios en esquema SQL del resto del código
    
    Todos los métodos son asincrónicos (async) para no bloquear el event loop.
    """
    
    # ========================================================================
    # OPERACIONES CRUD BÁSICAS DE OBJETO ASTRONÓMICO
    # ========================================================================
    
    async def crear_objeto(
        self,
        nombre: str,
        descripcion: Optional[str] = None
    ) -> ObjetoAstronomico:
        """
        Crea un nuevo objeto astronómico genérico en la base de datos.
        
        Inserta una fila en Objeto_Astronomico con nombre y descripción científica.
        Se utiliza como base para especializaciones (galaxias, estrellas, etc.).
        
        Args:
            nombre: Nombre del objeto (ej: "Vía Láctea", "Proxima Centauri")
            descripcion: Descripción científica opcional del objeto
        
        Returns:
            ObjetoAstronomico con id_objeto asignado por la BD
        
        Raises:
            ValueError: Si el nombre está vacío
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioObjetos()
            >>> objeto = await repo.crear_objeto(
            ...     nombre="Andromeda",
            ...     descripcion="Galaxia espiral a 2.5 millones de años luz"
            ... )
            >>> objeto.id_objeto  # Asignado por BD
            1
        """
        if not nombre or not nombre.strip():
            raise ValueError("El nombre del objeto no puede estar vacío")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_objeto = await conexion.fetchval(
                    """
                    INSERT INTO Objeto_Astronomico (nombre, descripcion_cientifica)
                    VALUES ($1, $2)
                    RETURNING id_objeto
                    """,
                    nombre.strip(),
                    descripcion.strip() if descripcion else None
                )
                
                return ObjetoAstronomico(
                    id_objeto=id_objeto,
                    nombre=nombre.strip(),
                    descripcion_cientifica=descripcion.strip() if descripcion else None
                )
        
        except Exception as e:
            raise RuntimeError(
                f"Error al crear objeto astronómico '{nombre}': {e}"
            ) from e
    
    async def obtener_objeto_por_id(
        self,
        id_objeto: int
    ) -> Optional[ObjetoAstronomico]:
        """
        Recupera un objeto astronómico por su identificador único.
        
        Args:
            id_objeto: ID del objeto a buscar
        
        Returns:
            ObjetoAstronomico si existe, None en caso contrario
        
        Raises:
            ValueError: Si id_objeto no es positivo
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioObjetos()
            >>> objeto = await repo.obtener_objeto_por_id(1)
            >>> objeto.nombre if objeto else "No encontrado"
            'Vía Láctea'
        """
        if not isinstance(id_objeto, int) or id_objeto <= 0:
            raise ValueError("id_objeto debe ser un entero positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                fila = await conexion.fetchrow(
                    """
                    SELECT id_objeto, nombre, descripcion_cientifica
                    FROM Objeto_Astronomico
                    WHERE id_objeto = $1
                    """,
                    id_objeto
                )
                
                if not fila:
                    return None
                
                return ObjetoAstronomico(
                    id_objeto=fila['id_objeto'],
                    nombre=fila['nombre'],
                    descripcion_cientifica=fila['descripcion_cientifica']
                )
        
        except Exception as e:
            raise RuntimeError(
                f"Error al obtener objeto con id {id_objeto}: {e}"
            ) from e
    
    async def obtener_objeto_por_nombre(
        self,
        nombre: str
    ) -> Optional[ObjetoAstronomico]:
        """
        Recupera un objeto astronómico por su nombre.
        
        Nota: Si hay múltiples objetos con el mismo nombre, retorna el primero encontrado.
        
        Args:
            nombre: Nombre del objeto a buscar
        
        Returns:
            ObjetoAstronomico si existe, None en caso contrario
        
        Raises:
            ValueError: Si el nombre está vacío
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioObjetos()
            >>> objeto = await repo.obtener_objeto_por_nombre("Vía Láctea")
            >>> objeto.id_objeto if objeto else None
            1
        """
        if not nombre or not nombre.strip():
            raise ValueError("El nombre no puede estar vacío")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                fila = await conexion.fetchrow(
                    """
                    SELECT id_objeto, nombre, descripcion_cientifica
                    FROM Objeto_Astronomico
                    WHERE nombre ILIKE $1
                    LIMIT 1
                    """,
                    nombre.strip()
                )
                
                if not fila:
                    return None
                
                return ObjetoAstronomico(
                    id_objeto=fila['id_objeto'],
                    nombre=fila['nombre'],
                    descripcion_cientifica=fila['descripcion_cientifica']
                )
        
        except Exception as e:
            raise RuntimeError(
                f"Error al buscar objeto por nombre '{nombre}': {e}"
            ) from e
    
    async def actualizar_descripcion(
        self,
        id_objeto: int,
        nueva_descripcion: str
    ) -> bool:
        """
        Actualiza la descripción científica de un objeto.
        
        Args:
            id_objeto: ID del objeto a actualizar
            nueva_descripcion: Nueva descripción científica
        
        Returns:
            True si se actualizó, False si el objeto no existe
        
        Raises:
            ValueError: Si el id_objeto no es válido
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioObjetos()
            >>> actualizado = await repo.actualizar_descripcion(1, "Nueva descripción")
            >>> actualizado
            True
        """
        if not isinstance(id_objeto, int) or id_objeto <= 0:
            raise ValueError("id_objeto debe ser un entero positivo")
        
        if not nueva_descripcion or not nueva_descripcion.strip():
            raise ValueError("La descripción no puede estar vacía")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                filas_afectadas = await conexion.execute(
                    """
                    UPDATE Objeto_Astronomico
                    SET descripcion_cientifica = $1
                    WHERE id_objeto = $2
                    """,
                    nueva_descripcion.strip(),
                    id_objeto
                )
                
                return filas_afectadas != "UPDATE 0"
        
        except Exception as e:
            raise RuntimeError(
                f"Error al actualizar descripción del objeto {id_objeto}: {e}"
            ) from e
    
    async def eliminar_objeto(self, id_objeto: int) -> bool:
        """
        Elimina un objeto astronómico y sus dependencias en cascada.
        
        La cascada se maneja mediante ON DELETE CASCADE en el esquema de BD.
        Se eliminan automáticamente: subtipos, documentos relacionados, embeddings, etc.
        
        Args:
            id_objeto: ID del objeto a eliminar
        
        Returns:
            True si se eliminó, False si no existe
        
        Raises:
            ValueError: Si id_objeto no es válido
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioObjetos()
            >>> eliminado = await repo.eliminar_objeto(1)
            >>> eliminado
            True
        """
        if not isinstance(id_objeto, int) or id_objeto <= 0:
            raise ValueError("id_objeto debe ser un entero positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                filas_afectadas = await conexion.execute(
                    """
                    DELETE FROM Objeto_Astronomico
                    WHERE id_objeto = $1
                    """,
                    id_objeto
                )
                
                return filas_afectadas != "DELETE 0"
        
        except Exception as e:
            raise RuntimeError(
                f"Error al eliminar objeto {id_objeto}: {e}"
            ) from e
    
    # ========================================================================
    # OPERACIONES ESPECÍFICAS DE PLANETAS
    # ========================================================================
    
    async def crear_planeta(self, datos: Planeta) -> Planeta:
        """
        Crea un nuevo planeta en la base de datos.
        
        Inserta en Objeto_Astronomico y Planeta con todos los atributos:
        tipo, sistema, masa, temperatura.
        
        Args:
            datos: Objeto Planeta con los datos a crear
        
        Returns:
            Planeta creado con id_objeto asignado
        
        Raises:
            ValueError: Si los datos son inválidos
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioObjetos()
            >>> planeta = Planeta(
            ...     id_objeto=999,  # Ignorado, asignado por BD
            ...     nombre="Tierra",
            ...     descripcion_cientifica="Planeta habitable",
            ...     id_tipo_planeta=1,
            ...     id_sistema=1,
            ...     masa=1.0,
            ...     temperatura=288
            ... )
            >>> nuevo = await repo.crear_planeta(planeta)
        """
        if not datos.nombre or not datos.nombre.strip():
            raise ValueError("El nombre del planeta no puede estar vacío")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                # Insertar objeto astronómico base
                id_objeto = await conexion.fetchval(
                    """
                    INSERT INTO Objeto_Astronomico (nombre, descripcion_cientifica)
                    VALUES ($1, $2)
                    RETURNING id_objeto
                    """,
                    datos.nombre.strip(),
                    datos.descripcion_cientifica.strip() if datos.descripcion_cientifica else None
                )
                
                # Insertar planeta
                await conexion.execute(
                    """
                    INSERT INTO Planeta (id_objeto, id_tipo_planeta, id_sistema, 
                                         masa, temperatura)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    id_objeto,
                    datos.id_tipo_planeta,
                    datos.id_sistema,
                    datos.masa,
                    datos.temperatura
                )
                
                # Retornar planeta con id asignado
                datos.id_objeto = id_objeto
                return datos
        
        except Exception as e:
            raise RuntimeError(
                f"Error al crear planeta '{datos.nombre}': {e}"
            ) from e
    
    async def crear_galaxia(self, datos: Galaxia) -> Galaxia:
        """Crea una nueva galaxia en Objeto_Astronomico y Galaxia."""
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_objeto = await conexion.fetchval(
                    """
                    INSERT INTO Objeto_Astronomico (nombre, descripcion_cientifica)
                    VALUES ($1, $2)
                    RETURNING id_objeto
                    """,
                    datos.nombre.strip(),
                    datos.descripcion_cientifica.strip() if datos.descripcion_cientifica else None
                )
                await conexion.execute(
                    """
                    INSERT INTO Galaxia (id_objeto, id_tipo_galaxia, distancia)
                    VALUES ($1, $2, $3)
                    """,
                    id_objeto,
                    datos.id_tipo_galaxia,
                    datos.distancia
                )
                return Galaxia(
                    id_objeto=id_objeto,
                    nombre=datos.nombre,
                    descripcion_cientifica=datos.descripcion_cientifica,
                    id_tipo_galaxia=datos.id_tipo_galaxia,
                    distancia=datos.distancia
                )
        except Exception as e:
            raise RuntimeError(f"Error al crear galaxia '{datos.nombre}': {e}") from e

    async def crear_sistema_estelar(self, datos: SistemaEstelar) -> SistemaEstelar:
        """Crea un nuevo sistema estelar en Objeto_Astronomico y Sistema_Estelar."""
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_objeto = await conexion.fetchval(
                    """
                    INSERT INTO Objeto_Astronomico (nombre, descripcion_cientifica)
                    VALUES ($1, $2)
                    RETURNING id_objeto
                    """,
                    datos.nombre.strip(),
                    datos.descripcion_cientifica.strip() if datos.descripcion_cientifica else None
                )
                await conexion.execute(
                    """
                    INSERT INTO Sistema_Estelar (id_objeto, id_galaxia)
                    VALUES ($1, $2)
                    """,
                    id_objeto,
                    datos.id_galaxia
                )
                return SistemaEstelar(
                    id_objeto=id_objeto,
                    nombre=datos.nombre,
                    descripcion_cientifica=datos.descripcion_cientifica,
                    id_galaxia=datos.id_galaxia
                )
        except Exception as e:
            raise RuntimeError(f"Error al crear sistema estelar '{datos.nombre}': {e}") from e

    async def crear_estrella(self, datos: Estrella) -> Estrella:
        """Crea una nueva estrella en Objeto_Astronomico y Estrella."""
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_objeto = await conexion.fetchval(
                    """
                    INSERT INTO Objeto_Astronomico (nombre, descripcion_cientifica)
                    VALUES ($1, $2)
                    RETURNING id_objeto
                    """,
                    datos.nombre.strip(),
                    datos.descripcion_cientifica.strip() if datos.descripcion_cientifica else None
                )
                await conexion.execute(
                    """
                    INSERT INTO Estrella (id_objeto, id_tipo_estrella, id_sistema, masa, temperatura)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    id_objeto,
                    datos.id_tipo_estrella,
                    datos.id_sistema,
                    datos.masa,
                    datos.temperatura
                )
                return Estrella(
                    id_objeto=id_objeto,
                    nombre=datos.nombre,
                    descripcion_cientifica=datos.descripcion_cientifica,
                    id_tipo_estrella=datos.id_tipo_estrella,
                    id_sistema=datos.id_sistema,
                    masa=datos.masa,
                    temperatura=datos.temperatura
                )
        except Exception as e:
            raise RuntimeError(f"Error al crear estrella '{datos.nombre}': {e}") from e

    async def crear_luna(self, datos: Luna) -> Luna:
        """Crea una nueva luna en Objeto_Astronomico y Luna."""
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_objeto = await conexion.fetchval(
                    """
                    INSERT INTO Objeto_Astronomico (nombre, descripcion_cientifica)
                    VALUES ($1, $2)
                    RETURNING id_objeto
                    """,
                    datos.nombre.strip(),
                    datos.descripcion_cientifica.strip() if datos.descripcion_cientifica else None
                )
                await conexion.execute(
                    """
                    INSERT INTO Luna (id_objeto, id_planeta, radio)
                    VALUES ($1, $2, $3)
                    """,
                    id_objeto,
                    datos.id_planeta,
                    datos.radio
                )
                return Luna(
                    id_objeto=id_objeto,
                    nombre=datos.nombre,
                    descripcion_cientifica=datos.descripcion_cientifica,
                    id_planeta=datos.id_planeta,
                    radio=datos.radio
                )
        except Exception as e:
            raise RuntimeError(f"Error al crear luna '{datos.nombre}': {e}") from e

    async def listar_planetas_por_habitabilidad(
        self,
        puntaje_minimo: float = 0.0
    ) -> List[Planeta]:
        """
        Lista planetas filtrados por puntaje mínimo de habitabilidad.
        
        Realiza JOIN con Evaluacion_Habitabilidad para retornar solo planetas
        evaluados con puntaje >= puntaje_minimo.
        
        Args:
            puntaje_minimo: Puntaje mínimo de habitabilidad (0.0 - 1.0)
        
        Returns:
            Lista de Planeta ordenados por puntaje descendente
        
        Raises:
            ValueError: Si puntaje_minimo no está en rango válido
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioObjetos()
            >>> habitables = await repo.listar_planetas_por_habitabilidad(0.7)
            >>> len(habitables)
            5
        """
        if not (0.0 <= puntaje_minimo <= 1.0):
            raise ValueError("puntaje_minimo debe estar entre 0.0 y 1.0")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                filas = await conexion.fetch(
                    """
                    SELECT p.id_objeto, o.nombre, o.descripcion_cientifica,
                           p.id_tipo_planeta, p.id_sistema, p.masa,
                           p.temperatura, eh.puntaje
                    FROM Planeta p
                    JOIN Objeto_Astronomico o ON p.id_objeto = o.id_objeto
                    JOIN Evaluacion_Habitabilidad eh ON p.id_objeto = eh.id_planeta
                    WHERE eh.puntaje >= $1
                    ORDER BY eh.puntaje DESC
                    """,
                    puntaje_minimo
                )
                
                planetas = []
                for fila in filas:
                    planetas.append(
                        Planeta(
                            id_objeto=fila['id_objeto'],
                            nombre=fila['nombre'],
                            descripcion_cientifica=fila['descripcion_cientifica'],
                            id_tipo_planeta=fila['id_tipo_planeta'],
                            id_sistema=fila['id_sistema'],
                            masa=fila['masa'],
                            temperatura=fila['temperatura']
                        )
                    )
                
                return planetas
        
        except Exception as e:
            raise RuntimeError(
                f"Error al listar planetas por habitabilidad: {e}"
            ) from e
    
    # ========================================================================
    # OPERACIONES DE CARACTERÍSTICAS AMBIENTALES
    # ========================================================================
    
    async def obtener_caracteristicas_ambientales(
        self,
        id_planeta: int
    ) -> List[CaracteristicaAmbiental]:
        """
        Recupera todas las características ambientales de un planeta.
        
        Retorna el conjunto de observaciones (presión, composición, humedad, etc.)
        registradas para evaluación de habitabilidad.
        
        Args:
            id_planeta: ID del planeta
        
        Returns:
            Lista de CaracteristicaAmbiental del planeta
        
        Raises:
            ValueError: Si id_planeta no es válido
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioObjetos()
            >>> caracteristicas = await repo.obtener_caracteristicas_ambientales(1)
            >>> len(caracteristicas)
            3
        """
        if not isinstance(id_planeta, int) or id_planeta <= 0:
            raise ValueError("id_planeta debe ser un entero positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                filas = await conexion.fetch(
                    """
                    SELECT id_caracteristica, id_planeta, tipo, valor
                    FROM Caracteristica_Ambiental
                    WHERE id_planeta = $1
                    ORDER BY tipo
                    """,
                    id_planeta
                )
                
                caracteristicas = []
                for fila in filas:
                    caracteristicas.append(
                        CaracteristicaAmbiental(
                            id_caracteristica=fila['id_caracteristica'],
                            id_planeta=fila['id_planeta'],
                            tipo=fila['tipo'],
                            valor=fila['valor']
                        )
                    )
                
                return caracteristicas
        
        except Exception as e:
            raise RuntimeError(
                f"Error al obtener características del planeta {id_planeta}: {e}"
            ) from e
    
    async def agregar_caracteristica_ambiental(
        self,
        datos: CaracteristicaAmbiental
    ) -> CaracteristicaAmbiental:
        """
        Agrega una característica ambiental a un planeta.
        
        Inserta una observación nueva (ej: "agua_liquida" = "presente") para
        el análisis de habitabilidad.
        
        Args:
            datos: CaracteristicaAmbiental con los datos a insertar
        
        Returns:
            CaracteristicaAmbiental creada con id asignado
        
        Raises:
            ValueError: Si los datos son inválidos
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioObjetos()
            >>> caracteristica = CaracteristicaAmbiental(
            ...     id_caracteristica=999,  # Ignorado
            ...     id_planeta=1,
            ...     tipo="oxigeno",
            ...     valor="21%"
            ... )
            >>> nueva = await repo.agregar_caracteristica_ambiental(caracteristica)
        """
        if not isinstance(datos.id_planeta, int) or datos.id_planeta <= 0:
            raise ValueError("id_planeta debe ser un entero positivo")
        
        if not datos.tipo or not datos.tipo.strip():
            raise ValueError("El tipo de característica no puede estar vacío")
        
        if not datos.valor or not datos.valor.strip():
            raise ValueError("El valor no puede estar vacío")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_caracteristica = await conexion.fetchval(
                    """
                    INSERT INTO Caracteristica_Ambiental (id_planeta, tipo, valor)
                    VALUES ($1, $2, $3)
                    RETURNING id_caracteristica
                    """,
                    datos.id_planeta,
                    datos.tipo.strip(),
                    datos.valor.strip()
                )
                
                datos.id_caracteristica = id_caracteristica
                return datos
        
        except Exception as e:
            raise RuntimeError(
                f"Error al agregar característica ambiental: {e}"
            ) from e