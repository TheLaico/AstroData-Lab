"""
Módulo de repositorio para documentos e imágenes en AstroData Lab.

Proporciona la capa de acceso a datos para documentos científicos, imágenes astronómicas
y sus embeddings vectoriales. Gestiona tanto la capa relacional (tablas Documento, Imagen)
como la capa vectorial (pgvector con Embedding_Texto e Embedding_Imagen).
"""

from typing import List, Optional, Dict, Any
from database.conexion import conexion_bd
from models.resultado import Documento, Imagen


class RepositorioDocumentos:
    """
    Repositorio para gestionar documentos, imágenes y sus embeddings.
    
    Encapsula operaciones CRUD sobre Documento e Imagen, así como la persistencia
    y búsqueda de embeddings vectoriales en pgvector. Diseñado con OCP para
    permitir extensión sin modificación.
    
    Gestiona dos tipos de datos:
    1. Documentos: textos científicos indexados por chunks, cada uno con embedding
    2. Imágenes: archivos visuales astronómicos con embeddings CLIP
    
    Las búsquedas vectoriales utilizan el operador <=> de pgvector (distancia L2)
    para encontrar embeddings más similares a una consulta.
    """
    
    # ========================================================================
    # OPERACIONES CRUD DE DOCUMENTO
    # ========================================================================
    
    async def crear_documento(self, datos: Documento) -> Documento:
        """
        Crea un nuevo documento científico en la base de datos.
        
        Inserta en tabla Documento todos los metadatos: título, idioma, fecha,
        fuente y contenido textual completo.
        
        SQL utilizado:
            INSERT INTO Documento (titulo, idioma, fecha, fuente, contenido_texto, id_objeto)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id_doc
        
        Args:
            datos: Objeto Documento con los datos a insertar
        
        Returns:
            Documento creado con id_doc asignado por la BD
        
        Raises:
            ValueError: Si el título está vacío
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioDocumentos()
            >>> doc = Documento(
            ...     id_doc=999,  # Ignorado
            ...     titulo="Observaciones de la Vía Láctea",
            ...     idioma="es",
            ...     fecha=datetime.now(),
            ...     fuente="NASA",
            ...     contenido_texto="...",
            ...     id_objeto=1
            ... )
            >>> nuevo = await repo.crear_documento(doc)
        """
        if not datos.titulo or not datos.titulo.strip():
            raise ValueError("El título del documento no puede estar vacío")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_doc = await conexion.fetchval(
                    """
                    INSERT INTO Documento (titulo, idioma, fecha, fuente, contenido_texto, id_objeto)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id_doc
                    """,
                    datos.titulo.strip(),
                    datos.idioma,
                    datos.fecha,
                    datos.fuente,
                    datos.contenido_texto,
                    datos.id_objeto
                )
                
                datos.id_doc = id_doc
                return datos
        
        except Exception as e:
            raise RuntimeError(
                f"Error al crear documento '{datos.titulo}': {e}"
            ) from e
    
    async def obtener_documento_por_id(self, id_doc: int) -> Optional[Documento]:
        """
        Recupera un documento por su identificador único.
        
        SQL utilizado:
            SELECT id_doc, titulo, idioma, fecha, fuente, contenido_texto, id_objeto
            FROM Documento
            WHERE id_doc = $1
        
        Args:
            id_doc: ID del documento a buscar
        
        Returns:
            Documento si existe, None en caso contrario
        
        Raises:
            ValueError: Si id_doc no es positivo
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioDocumentos()
            >>> doc = await repo.obtener_documento_por_id(1)
            >>> doc.titulo if doc else "No encontrado"
        """
        if not isinstance(id_doc, int) or id_doc <= 0:
            raise ValueError("id_doc debe ser un entero positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                fila = await conexion.fetchrow(
                    """
                    SELECT id_doc, titulo, idioma, fecha, fuente, contenido_texto, id_objeto
                    FROM Documento
                    WHERE id_doc = $1
                    """,
                    id_doc
                )
                
                if not fila:
                    return None
                
                return Documento(
                    id_doc=fila['id_doc'],
                    titulo=fila['titulo'],
                    idioma=fila['idioma'],
                    fecha=fila['fecha'],
                    fuente=fila['fuente'],
                    contenido_texto=fila['contenido_texto'],
                    id_objeto=fila['id_objeto']
                )
        
        except Exception as e:
            raise RuntimeError(
                f"Error al obtener documento con id {id_doc}: {e}"
            ) from e
    
    async def listar_documentos_por_objeto(
        self,
        id_objeto: int
    ) -> List[Documento]:
        """
        Lista todos los documentos asociados a un objeto astronómico específico.
        
        SQL utilizado:
            SELECT id_doc, titulo, idioma, fecha, fuente, contenido_texto, id_objeto
            FROM Documento
            WHERE id_objeto = $1
            ORDER BY fecha DESC
        
        Args:
            id_objeto: ID del objeto astronómico
        
        Returns:
            Lista de Documento ordenados por fecha descendente
        
        Raises:
            ValueError: Si id_objeto no es positivo
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioDocumentos()
            >>> docs = await repo.listar_documentos_por_objeto(1)
            >>> len(docs)
            3
        """
        if not isinstance(id_objeto, int) or id_objeto <= 0:
            raise ValueError("id_objeto debe ser un entero positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                filas = await conexion.fetch(
                    """
                    SELECT id_doc, titulo, idioma, fecha, fuente, contenido_texto, id_objeto
                    FROM Documento
                    WHERE id_objeto = $1
                    ORDER BY fecha DESC
                    """,
                    id_objeto
                )
                
                documentos = []
                for fila in filas:
                    documentos.append(
                        Documento(
                            id_doc=fila['id_doc'],
                            titulo=fila['titulo'],
                            idioma=fila['idioma'],
                            fecha=fila['fecha'],
                            fuente=fila['fuente'],
                            contenido_texto=fila['contenido_texto'],
                            id_objeto=fila['id_objeto']
                        )
                    )
                
                return documentos
        
        except Exception as e:
            raise RuntimeError(
                f"Error al listar documentos del objeto {id_objeto}: {e}"
            ) from e
    
    # ========================================================================
    # OPERACIONES DE EMBEDDINGS DE TEXTO
    # ========================================================================
    
    async def guardar_embedding_texto(
        self,
        id_doc: int,
        chunk_id: int,
        estrategia: str,
        vector: List[float],
        modelo: str
    ) -> int:
        """
        Guarda el embedding vectorial de un chunk de texto.
        
        Inserta en tabla Embedding_Texto con el vector normalizado y metadatos
        sobre la estrategia de chunking y modelo usado.
        
        SQL utilizado:
            INSERT INTO Embedding_Texto (id_doc, chunk_id, estrategia_chunking, vector, modelo)
            VALUES ($1, $2, $3, $4::vector, $5)
            RETURNING id
        
        Args:
            id_doc: ID del documento origen
            chunk_id: Número secuencial del chunk dentro del documento
            estrategia: Estrategia de chunking ('fixed', 'sentence', 'semantic')
            vector: Lista de flotantes del embedding (384 dimensiones para MiniLM)
            modelo: Nombre del modelo de embeddings usado
        
        Returns:
            ID del embedding creado
        
        Raises:
            ValueError: Si los parámetros son inválidos
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioDocumentos()
            >>> embedding_id = await repo.guardar_embedding_texto(
            ...     id_doc=1,
            ...     chunk_id=0,
            ...     estrategia='sentence',
            ...     vector=[0.1, 0.2, ...],  # 384 valores
            ...     modelo='all-MiniLM-L6-v2'
            ... )
        """
        if id_doc <= 0 or chunk_id < 0:
            raise ValueError("id_doc debe ser positivo y chunk_id >= 0")
        
        if estrategia not in ('fixed', 'sentence', 'semantic'):
            raise ValueError("estrategia debe ser 'fixed', 'sentence' o 'semantic'")
        
        if not vector or len(vector) == 0:
            raise ValueError("El vector no puede estar vacío")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                # Convertir lista de Python a formato pgvector
                vector_str = f"[{','.join(str(v) for v in vector)}]"
                
                embedding_id = await conexion.fetchval(
                    """
                    INSERT INTO Embedding_Texto (id_doc, chunk_id, estrategia_chunking, vector, modelo)
                    VALUES ($1, $2, $3, $4::vector, $5)
                    RETURNING id
                    """,
                    id_doc,
                    chunk_id,
                    estrategia,
                    vector_str,
                    modelo
                )
                
                return embedding_id
        
        except Exception as e:
            raise RuntimeError(
                f"Error al guardar embedding de texto para doc {id_doc}: {e}"
            ) from e
    
    async def buscar_chunks_similares(
        self,
        vector_consulta: List[float],
        top_k: int = 5,
        estrategia: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca los chunks de texto más similares a un vector de consulta.
        
        Utiliza el operador <=> de pgvector (distancia coseno) para encontrar
        los embeddings más cercanos. Opcionalmente filtra por estrategia de chunking.
        
        SQL utilizado (sin filtro de estrategia):
            SELECT d.titulo, e.chunk_id, e.estrategia_chunking,
                   1 - (e.vector <=> $1::vector) AS similitud
            FROM Embedding_Texto e
            JOIN Documento d ON e.id_doc = d.id_doc
            ORDER BY e.vector <=> $1::vector
            LIMIT $2
        
        SQL utilizado (con filtro de estrategia):
            SELECT d.titulo, e.chunk_id, e.estrategia_chunking,
                   1 - (e.vector <=> $1::vector) AS similitud
            FROM Embedding_Texto e
            JOIN Documento d ON e.id_doc = d.id_doc
            WHERE e.estrategia_chunking = $2
            ORDER BY e.vector <=> $1::vector
            LIMIT $3
        
        Args:
            vector_consulta: Vector de embedding de la consulta (384 dimensiones)
            top_k: Número máximo de resultados a retornar (default: 5)
            estrategia: Filtrar por estrategia ('fixed', 'sentence', 'semantic') u omitir
        
        Returns:
            Lista de dicts con keys: 'titulo', 'chunk_id', 'estrategia_chunking', 'similitud'
            Ordenados por similitud descendente.
        
        Raises:
            ValueError: Si los parámetros son inválidos
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioDocumentos()
            >>> resultados = await repo.buscar_chunks_similares(
            ...     vector_consulta=[0.1, 0.2, ...],  # 384 valores
            ...     top_k=5,
            ...     estrategia='sentence'
            ... )
            >>> resultados[0]['similitud']  # Entre 0.0 y 1.0
            0.87
        """
        if not vector_consulta or len(vector_consulta) == 0:
            raise ValueError("El vector de consulta no puede estar vacío")
        
        if top_k <= 0:
            raise ValueError("top_k debe ser positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                vector_str = f"[{','.join(str(v) for v in vector_consulta)}]"
                
                if estrategia:
                    if estrategia not in ('fixed', 'sentence', 'semantic'):
                        raise ValueError("estrategia inválida")
                    
                    filas = await conexion.fetch(
                        """
                        SELECT d.titulo, e.chunk_id, e.estrategia_chunking,
                               1 - (e.vector <=> $1::vector) AS similitud
                        FROM Embedding_Texto e
                        JOIN Documento d ON e.id_doc = d.id_doc
                        WHERE e.estrategia_chunking = $2
                        ORDER BY e.vector <=> $1::vector
                        LIMIT $3
                        """,
                        vector_str,
                        estrategia,
                        top_k
                    )
                else:
                    filas = await conexion.fetch(
                        """
                        SELECT d.titulo, e.chunk_id, e.estrategia_chunking,
                               1 - (e.vector <=> $1::vector) AS similitud
                        FROM Embedding_Texto e
                        JOIN Documento d ON e.id_doc = d.id_doc
                        ORDER BY e.vector <=> $1::vector
                        LIMIT $2
                        """,
                        vector_str,
                        top_k
                    )
                
                resultados = []
                for fila in filas:
                    resultados.append({
                        'titulo': fila['titulo'],
                        'chunk_id': fila['chunk_id'],
                        'estrategia_chunking': fila['estrategia_chunking'],
                        'similitud': float(fila['similitud'])
                    })
                
                return resultados
        
        except Exception as e:
            raise RuntimeError(
                f"Error al buscar chunks similares: {e}"
            ) from e
    
    # ========================================================================
    # OPERACIONES CRUD DE IMAGEN
    # ========================================================================
    
    async def crear_imagen(self, datos: Imagen) -> Imagen:
        """
        Crea un nuevo registro de imagen astronómica.
        
        SQL utilizado:
            INSERT INTO Imagen (ruta_archivo, descripcion, etiquetas, id_doc)
            VALUES ($1, $2, $3, $4)
            RETURNING id_imagen
        
        Args:
            datos: Objeto Imagen con los datos a insertar
        
        Returns:
            Imagen creada con id_imagen asignado
        
        Raises:
            ValueError: Si la ruta está vacía
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioDocumentos()
            >>> imagen = Imagen(
            ...     id_imagen=999,  # Ignorado
            ...     ruta_archivo="/datos/galaxias/m31.jpg",
            ...     descripcion="Galaxia de Andromeda",
            ...     etiquetas=["galaxia", "espiral"],
            ...     id_doc=1
            ... )
            >>> nueva = await repo.crear_imagen(imagen)
        """
        if not datos.ruta_archivo or not datos.ruta_archivo.strip():
            raise ValueError("La ruta del archivo no puede estar vacía")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_imagen = await conexion.fetchval(
                    """
                    INSERT INTO Imagen (ruta_archivo, descripcion, etiquetas, id_doc)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id_imagen
                    """,
                    datos.ruta_archivo.strip(),
                    datos.descripcion,
                    datos.etiquetas,
                    datos.id_doc
                )
                
                datos.id_imagen = id_imagen
                return datos
        
        except Exception as e:
            raise RuntimeError(
                f"Error al crear imagen: {e}"
            ) from e
    
    # ========================================================================
    # OPERACIONES DE EMBEDDINGS DE IMAGEN
    # ========================================================================
    
    async def guardar_embedding_imagen(
        self,
        id_imagen: int,
        vector: List[float],
        modelo: str
    ) -> int:
        """
        Guarda el embedding vectorial de una imagen astronómica.
        
        Inserta en tabla Embedding_Imagen el vector CLIP (512 dimensiones)
        de la imagen para búsqueda visual por similitud.
        
        SQL utilizado:
            INSERT INTO Embedding_Imagen (id_imagen, vector, modelo)
            VALUES ($1, $2::vector, $3)
            RETURNING id
        
        Args:
            id_imagen: ID de la imagen origen
            vector: Lista de flotantes del embedding (512 dimensiones para CLIP)
            modelo: Nombre del modelo CLIP usado
        
        Returns:
            ID del embedding creado
        
        Raises:
            ValueError: Si los parámetros son inválidos
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioDocumentos()
            >>> embedding_id = await repo.guardar_embedding_imagen(
            ...     id_imagen=1,
            ...     vector=[0.1, 0.2, ...],  # 512 valores
            ...     modelo='openai/clip-vit-base-patch32'
            ... )
        """
        if id_imagen <= 0:
            raise ValueError("id_imagen debe ser positivo")
        
        if not vector or len(vector) == 0:
            raise ValueError("El vector no puede estar vacío")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                vector_str = f"[{','.join(str(v) for v in vector)}]"
                
                embedding_id = await conexion.fetchval(
                    """
                    INSERT INTO Embedding_Imagen (id_imagen, vector, modelo)
                    VALUES ($1, $2::vector, $3)
                    RETURNING id
                    """,
                    id_imagen,
                    vector_str,
                    modelo
                )
                
                return embedding_id
        
        except Exception as e:
            raise RuntimeError(
                f"Error al guardar embedding de imagen {id_imagen}: {e}"
            ) from e
    
    async def buscar_imagenes_similares(
        self,
        vector_consulta: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Busca las imágenes más similares a un vector de consulta.
        
        Utiliza el operador <=> de pgvector (distancia L2) para encontrar
        embeddings CLIP más cercanos. Retorna metadatos y similitud.
        
        SQL utilizado:
            SELECT i.id_imagen, i.ruta_archivo, i.descripcion,
                   1 - (e.vector <=> $1::vector) AS similitud
            FROM Embedding_Imagen e
            JOIN Imagen i ON e.id_imagen = i.id_imagen
            ORDER BY e.vector <=> $1::vector
            LIMIT $2
        
        Args:
            vector_consulta: Vector de embedding (puede ser CLIP o descripción textual)
            top_k: Número máximo de resultados
        
        Returns:
            Lista de dicts con keys: 'id_imagen', 'ruta_archivo', 'descripcion', 'similitud'
        
        Raises:
            ValueError: Si los parámetros son inválidos
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioDocumentos()
            >>> imagenes = await repo.buscar_imagenes_similares(
            ...     vector_consulta=[0.1, 0.2, ...],  # 512 valores
            ...     top_k=3
            ... )
            >>> imagenes[0]['similitud']
            0.92
        """
        if not vector_consulta or len(vector_consulta) == 0:
            raise ValueError("El vector de consulta no puede estar vacío")
        
        if top_k <= 0:
            raise ValueError("top_k debe ser positivo")
        
        try:
            async with conexion_bd.obtener_conexion() as conexion:
                vector_str = f"[{','.join(str(v) for v in vector_consulta)}]"
                
                filas = await conexion.fetch(
                    """
                    SELECT i.id_imagen, i.ruta_archivo, i.descripcion,
                           1 - (e.vector <=> $1::vector) AS similitud
                    FROM Embedding_Imagen e
                    JOIN Imagen i ON e.id_imagen = i.id_imagen
                    ORDER BY e.vector <=> $1::vector
                    LIMIT $2
                    """,
                    vector_str,
                    top_k
                )
                
                resultados = []
                for fila in filas:
                    resultados.append({
                        'id_imagen': fila['id_imagen'],
                        'ruta_archivo': fila['ruta_archivo'],
                        'descripcion': fila['descripcion'],
                        'similitud': float(fila['similitud'])
                    })
                
                return resultados
        
        except Exception as e:
            raise RuntimeError(
                f"Error al buscar imágenes similares: {e}"
            ) from e
