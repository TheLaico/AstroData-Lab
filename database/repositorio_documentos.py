"""
Módulo de repositorio para documentos e imágenes en AstroData Lab.

Proporciona la capa de acceso a datos para documentos científicos, imágenes astronómicas
y sus embeddings vectoriales. Gestiona tanto la capa relacional (tablas Documento, Imagen)
como la capa vectorial (pgvector con Embedding_Texto e Embedding_Imagen).
"""

from typing import List, Optional
from database.conexion import conexion_bd
from models.documento import Documento, Imagen


class RepositorioDocumentos:
    """
    Repositorio para gestionar documentos, imágenes y sus embeddings.
    
    Encapsula operaciones CRUD sobre Documento e Imagen, así como la persistencia
    y búsqueda de embeddings vectoriales en pgvector. Diseñado con OCP para
    permitir extensión sin modificación.
    
    Gestiona dos tipos de datos:
    1. Documentos: textos científicos indexados por chunks, cada uno con embedding
    2. Imágenes: archivos visuales astronómicos con embeddings CLIP
    
    Las búsquedas vectoriales utilizan el operador <=> de pgvector (distancia coseno)
    para encontrar embeddings más similares a una consulta.
    """

    # OPERACIONES CRUD DE DOCUMENTO

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

    # OPERACIONES CRUD DE IMAGEN

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

    # OPERACIONES DE EMBEDDINGS DE TEXTO (pgvector)

    async def guardar_embedding_texto(
        self,
        id_doc: int,
        chunk_id: int,
        vector: List[float],
        modelo: str,
        estrategia_chunking: str
    ) -> int:
        """
        Persiste el embedding vectorial de un chunk de texto en pgvector.
        
        Cada documento se divide en chunks antes de generar embeddings.
        Cada chunk tiene su propio vector almacenado en Embedding_Texto
        para permitir búsqueda semántica a nivel de fragmento.
        
        SQL utilizado:
            INSERT INTO Embedding_Texto
                (id_doc, chunk_id, vector, modelo, estrategia_chunking)
            VALUES ($1, $2, $3::vector, $4, $5)
            RETURNING id_embedding
        
        Args:
            id_doc: ID del documento al que pertenece el chunk
            chunk_id: Número de chunk dentro del documento (empieza en 0)
            vector: Embedding numérico generado por el modelo (ej: 384 dims)
            modelo: Nombre del modelo usado (ej: 'sentence-transformers/all-MiniLM-L6-v2')
            estrategia_chunking: Estrategia usada para dividir el texto
                                 (ej: 'fixed', 'sentence', 'paragraph')
        
        Returns:
            id_embedding asignado por la base de datos
        
        Raises:
            ValueError: Si el vector está vacío, id_doc no es positivo,
                        o chunk_id es negativo
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioDocumentos()
            >>> id_emb = await repo.guardar_embedding_texto(
            ...     id_doc=1,
            ...     chunk_id=0,
            ...     vector=[0.12, -0.45, 0.88, ...],  # 384 valores
            ...     modelo="sentence-transformers/all-MiniLM-L6-v2",
            ...     estrategia_chunking="sentence"
            ... )
        """
        if not isinstance(id_doc, int) or id_doc <= 0:
            raise ValueError("id_doc debe ser un entero positivo")
        if not isinstance(chunk_id, int) or chunk_id < 0:
            raise ValueError("chunk_id debe ser un entero no negativo")
        if not vector:
            raise ValueError("El vector no puede estar vacío")

        # pgvector espera el vector como string '[x1, x2, ...]'
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_embedding = await conexion.fetchval(
                    """
                    INSERT INTO Embedding_Texto
                        (id_doc, chunk_id, vector, modelo, estrategia_chunking)
                    VALUES ($1, $2, $3::vector, $4, $5)
                    RETURNING id_embedding
                    """,
                    id_doc,
                    chunk_id,
                    vector_str,
                    modelo,
                    estrategia_chunking
                )

                return id_embedding

        except Exception as e:
            raise RuntimeError(
                f"Error al guardar embedding de texto para doc {id_doc}, "
                f"chunk {chunk_id}: {e}"
            ) from e

    async def buscar_chunks_similares(
        self,
        vector_consulta: List[float],
        top_k: int,
        estrategia_chunking: Optional[str] = None
    ) -> List[dict]:
        """
        Busca los chunks de texto más similares semánticamente a una consulta.
        
        Utiliza el operador <=> de pgvector (distancia coseno) sobre la tabla
        Embedding_Texto y hace JOIN con Documento para recuperar título y
        contenido. La similitud se normaliza como 1 - distancia_coseno,
        donde 1.0 = idéntico y 0.0 = sin relación.
        
        Si se especifica estrategia_chunking se filtra por esa estrategia,
        útil cuando el sistema usa múltiples estrategias en paralelo y se
        quiere comparar resultados o forzar una específica.
        
        SQL utilizado (sin filtro de estrategia):
            SELECT
                et.id_doc,
                d.titulo,
                et.chunk_id,
                et.estrategia_chunking   AS estrategia,
                1 - (et.vector <=> $1::vector) AS similitud,
                d.contenido_texto        AS contenido
            FROM Embedding_Texto et
            JOIN Documento d ON d.id_doc = et.id_doc
            ORDER BY et.vector <=> $1::vector
            LIMIT $2
        
        SQL utilizado (con filtro de estrategia):
            ... WHERE et.estrategia_chunking = $3
            ORDER BY et.vector <=> $1::vector
            LIMIT $2
        
        Args:
            vector_consulta: Embedding de la consulta del usuario
            top_k: Número máximo de chunks a retornar
            estrategia_chunking: Filtro opcional por estrategia de chunking
        
        Returns:
            Lista de dicts con las claves:
                - id_doc (int): ID del documento fuente
                - titulo (str): Título del documento
                - chunk_id (int): Índice del chunk dentro del documento
                - estrategia (str): Estrategia de chunking usada
                - similitud (float): Similitud coseno normalizada [0.0, 1.0]
                - contenido (str | None): Texto completo del documento
        
        Raises:
            ValueError: Si vector_consulta está vacío o top_k no es positivo
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioDocumentos()
            >>> resultados = await repo.buscar_chunks_similares(
            ...     vector_consulta=[0.12, -0.45, 0.88, ...],
            ...     top_k=5,
            ...     estrategia_chunking="sentence"
            ... )
            >>> for r in resultados:
            ...     print(r['titulo'], r['similitud'])
        """
        if not vector_consulta:
            raise ValueError("El vector de consulta no puede estar vacío")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k debe ser un entero positivo")

        vector_str = "[" + ",".join(str(v) for v in vector_consulta) + "]"

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                if estrategia_chunking is not None:
                    filas = await conexion.fetch(
                        """
                        SELECT
                            et.id_doc,
                            d.titulo,
                            et.chunk_id,
                            et.estrategia_chunking              AS estrategia,
                            1 - (et.vector <=> $1::vector)      AS similitud,
                            d.contenido_texto                   AS contenido
                        FROM Embedding_Texto et
                        JOIN Documento d ON d.id_doc = et.id_doc
                        WHERE et.estrategia_chunking = $3
                        ORDER BY et.vector <=> $1::vector
                        LIMIT $2
                        """,
                        vector_str,
                        top_k,
                        estrategia_chunking
                    )
                else:
                    filas = await conexion.fetch(
                        """
                        SELECT
                            et.id_doc,
                            d.titulo,
                            et.chunk_id,
                            et.estrategia_chunking              AS estrategia,
                            1 - (et.vector <=> $1::vector)      AS similitud,
                            d.contenido_texto                   AS contenido
                        FROM Embedding_Texto et
                        JOIN Documento d ON d.id_doc = et.id_doc
                        ORDER BY et.vector <=> $1::vector
                        LIMIT $2
                        """,
                        vector_str,
                        top_k
                    )

                return [
                    {
                        "id_doc":    fila["id_doc"],
                        "titulo":    fila["titulo"],
                        "chunk_id":  fila["chunk_id"],
                        "estrategia": fila["estrategia"],
                        "similitud": float(fila["similitud"]),
                        "contenido": fila["contenido"],
                    }
                    for fila in filas
                ]

        except Exception as e:
            raise RuntimeError(
                f"Error al buscar chunks similares: {e}"
            ) from e

    # OPERACIONES DE EMBEDDINGS DE IMAGEN (pgvector)

    async def guardar_embedding_imagen(
        self,
        id_imagen: int,
        vector: List[float],
        modelo: str
    ) -> int:
        """
        Persiste el embedding vectorial CLIP de una imagen en pgvector.
        
        Los embeddings de imagen permiten búsqueda semántica visual: dada una
        consulta de texto, su embedding puede compararse contra embeddings CLIP
        de imágenes para recuperar las más relevantes visualmente.
        
        SQL utilizado:
            INSERT INTO Embedding_Imagen (id_imagen, vector, modelo)
            VALUES ($1, $2::vector, $3)
            RETURNING id_embedding
        
        Args:
            id_imagen: ID de la imagen a la que pertenece el embedding
            vector: Embedding CLIP de la imagen (típicamente 512 dims con CLIP ViT-B/32)
            modelo: Nombre del modelo CLIP usado (ej: 'openai/clip-vit-base-patch32')
        
        Returns:
            id_embedding asignado por la base de datos
        
        Raises:
            ValueError: Si el vector está vacío o id_imagen no es positivo
            RuntimeError: Si hay error en la operación de BD
        
        Example:
            >>> repo = RepositorioDocumentos()
            >>> id_emb = await repo.guardar_embedding_imagen(
            ...     id_imagen=7,
            ...     vector=[0.03, 0.91, -0.22, ...],  # 512 valores CLIP
            ...     modelo="openai/clip-vit-base-patch32"
            ... )
        """
        if not isinstance(id_imagen, int) or id_imagen <= 0:
            raise ValueError("id_imagen debe ser un entero positivo")
        if not vector:
            raise ValueError("El vector no puede estar vacío")

        vector_str = "[" + ",".join(str(v) for v in vector) + "]"

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_embedding = await conexion.fetchval(
                    """
                    INSERT INTO Embedding_Imagen (id_imagen, vector, modelo)
                    VALUES ($1, $2::vector, $3)
                    RETURNING id_embedding
                    """,
                    id_imagen,
                    vector_str,
                    modelo
                )

                return id_embedding

        except Exception as e:
            raise RuntimeError(
                f"Error al guardar embedding de imagen {id_imagen}: {e}"
            ) from e