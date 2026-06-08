"""
Módulo de repositorio para documentos e imágenes en AstroData Lab.

Proporciona la capa de acceso a datos para documentos científicos, imágenes
astronómicas y sus embeddings vectoriales.

CAMBIO v2:
- guardar_embedding_texto ahora acepta y persiste contenido_chunk (texto
  real del fragmento) junto al vector.
- buscar_chunks_similares ya no hace JOIN con Documento para traer
  contenido_texto completo. Lee contenido_chunk directamente desde
  Embedding_Texto, que es el fragmento específico y pesa órdenes de
  magnitud menos que el documento completo.
"""

from typing import List, Optional
from database.conexion import conexion_bd
from models.documento_model import Documento
from models.imagen_model import Imagen


class RepositorioDocumentos:
    """
    Repositorio para gestionar documentos, imágenes y sus embeddings.

    Encapsula operaciones CRUD sobre Documento e Imagen, así como la
    persistencia y búsqueda de embeddings vectoriales en pgvector.

    Las búsquedas vectoriales utilizan el operador <=> de pgvector
    (distancia coseno) para encontrar embeddings más similares a una consulta.
    """

    # ------------------------------------------------------------------
    # OPERACIONES CRUD DE DOCUMENTO
    # ------------------------------------------------------------------

    async def crear_documento(self, datos: Documento) -> Documento:
        """
        Crea un nuevo documento científico en la base de datos.

        SQL:
            INSERT INTO Documento (titulo, idioma, fecha, fuente, contenido_texto, id_objeto)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id_doc

        Args:
            datos: Objeto Documento con los datos a insertar.

        Returns:
            Documento creado con id_doc asignado por la BD.

        Raises:
            ValueError: Si el título está vacío.
            RuntimeError: Si hay error en la operación de BD.
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

        SQL:
            SELECT id_doc, titulo, idioma, fecha, fuente, contenido_texto, id_objeto
            FROM Documento WHERE id_doc = $1

        Args:
            id_doc: ID del documento a buscar.

        Returns:
            Documento si existe, None en caso contrario.

        Raises:
            ValueError: Si id_doc no es positivo.
            RuntimeError: Si hay error en la operación de BD.
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

    async def listar_documentos_por_objeto(self, id_objeto: int) -> List[Documento]:
        """
        Lista todos los documentos asociados a un objeto astronómico.

        SQL:
            SELECT ... FROM Documento WHERE id_objeto = $1 ORDER BY fecha DESC

        Args:
            id_objeto: ID del objeto astronómico.

        Returns:
            Lista de Documento ordenados por fecha descendente.

        Raises:
            ValueError: Si id_objeto no es positivo.
            RuntimeError: Si hay error en la operación de BD.
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
                return [
                    Documento(
                        id_doc=fila['id_doc'],
                        titulo=fila['titulo'],
                        idioma=fila['idioma'],
                        fecha=fila['fecha'],
                        fuente=fila['fuente'],
                        contenido_texto=fila['contenido_texto'],
                        id_objeto=fila['id_objeto']
                    )
                    for fila in filas
                ]

        except Exception as e:
            raise RuntimeError(
                f"Error al listar documentos del objeto {id_objeto}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # OPERACIONES CRUD DE IMAGEN
    # ------------------------------------------------------------------

    async def crear_imagen(self, datos: Imagen) -> Imagen:
        """
        Crea un nuevo registro de imagen astronómica.

        SQL:
            INSERT INTO Imagen (ruta_archivo, descripcion, etiquetas, id_doc)
            VALUES ($1, $2, $3, $4)
            RETURNING id_imagen

        Args:
            datos: Objeto Imagen con los datos a insertar.

        Returns:
            Imagen creada con id_imagen asignado.

        Raises:
            ValueError: Si la ruta está vacía.
            RuntimeError: Si hay error en la operación de BD.
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
                    self._serializar_etiquetas(datos.etiquetas),
                    datos.id_doc
                )
                datos.id_imagen = id_imagen
                return datos

        except Exception as e:
            raise RuntimeError(f"Error al crear imagen: {e}") from e

    async def listar_imagenes_sin_embedding(self, limite: int = 50) -> List[Imagen]:
        """
        Lista imagenes que existen en Imagen pero no tienen Embedding_Imagen.
        """
        if not isinstance(limite, int) or limite <= 0:
            raise ValueError("limite debe ser un entero positivo")

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                filas = await conexion.fetch(
                    """
                    SELECT i.id_imagen, i.ruta_archivo, i.descripcion, i.etiquetas, i.id_doc
                    FROM Imagen i
                    LEFT JOIN Embedding_Imagen ei ON ei.id_imagen = i.id_imagen
                    WHERE ei.id_embedding IS NULL
                    ORDER BY i.id_imagen
                    LIMIT $1
                    """,
                    limite
                )
                return [
                    Imagen(
                        id_imagen=fila["id_imagen"],
                        ruta_archivo=fila["ruta_archivo"],
                        descripcion=fila["descripcion"],
                        etiquetas=self._deserializar_etiquetas(fila["etiquetas"]),
                        id_doc=fila["id_doc"],
                    )
                    for fila in filas
                ]
        except Exception as e:
            raise RuntimeError(f"Error al listar imagenes sin embedding: {e}") from e

    async def actualizar_imagen(self, datos: Imagen) -> Imagen:
        """
        Actualiza una imagen existente y devuelve sus datos actuales.
        """
        if not isinstance(datos.id_imagen, int) or datos.id_imagen <= 0:
            raise ValueError("id_imagen debe ser un entero positivo")
        if not datos.ruta_archivo or not datos.ruta_archivo.strip():
            raise ValueError("La ruta del archivo no puede estar vacia")

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                fila = await conexion.fetchrow(
                    """
                    UPDATE Imagen
                    SET ruta_archivo = $2,
                        descripcion = $3,
                        etiquetas = $4,
                        id_doc = $5
                    WHERE id_imagen = $1
                    RETURNING id_imagen, ruta_archivo, descripcion, etiquetas, id_doc
                    """,
                    datos.id_imagen,
                    datos.ruta_archivo.strip(),
                    datos.descripcion,
                    self._serializar_etiquetas(datos.etiquetas),
                    datos.id_doc,
                )
                if not fila:
                    raise ValueError(f"No existe imagen con id {datos.id_imagen}")
                return Imagen(
                    id_imagen=fila["id_imagen"],
                    ruta_archivo=fila["ruta_archivo"],
                    descripcion=fila["descripcion"],
                    etiquetas=self._deserializar_etiquetas(fila["etiquetas"]),
                    id_doc=fila["id_doc"],
                )
        except Exception as e:
            raise RuntimeError(f"Error al actualizar imagen {datos.id_imagen}: {e}") from e

    async def eliminar_embeddings_imagen(self, id_imagen: int) -> int:
        """
        Elimina embeddings existentes de una imagen para regenerarlos.
        """
        if not isinstance(id_imagen, int) or id_imagen <= 0:
            raise ValueError("id_imagen debe ser un entero positivo")

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                resultado = await conexion.execute(
                    "DELETE FROM Embedding_Imagen WHERE id_imagen = $1",
                    id_imagen,
                )
                return int(resultado.split()[-1])
        except Exception as e:
            raise RuntimeError(f"Error al eliminar embeddings de imagen {id_imagen}: {e}") from e

    async def eliminar_imagen(self, id_imagen: int) -> bool:
        """
        Elimina una imagen. Sus embeddings se eliminan por ON DELETE CASCADE.
        """
        if not isinstance(id_imagen, int) or id_imagen <= 0:
            raise ValueError("id_imagen debe ser un entero positivo")

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                resultado = await conexion.execute(
                    "DELETE FROM Imagen WHERE id_imagen = $1",
                    id_imagen,
                )
                return resultado != "DELETE 0"
        except Exception as e:
            raise RuntimeError(f"Error al eliminar imagen {id_imagen}: {e}") from e

    # ------------------------------------------------------------------
    # OPERACIONES DE EMBEDDINGS DE TEXTO (pgvector)
    # ------------------------------------------------------------------

    def _serializar_etiquetas(self, etiquetas: Optional[List[str]]) -> Optional[str]:
        if etiquetas is None:
            return None
        return ",".join(
            str(tag).strip()
            for tag in etiquetas
            if tag is not None and str(tag).strip()
        )

    def _deserializar_etiquetas(self, etiquetas: Optional[str]) -> Optional[List[str]]:
        if not etiquetas:
            return None
        return [tag.strip() for tag in etiquetas.split(",") if tag.strip()]

    async def guardar_embedding_texto(
        self,
        id_doc: int,
        chunk_id: int,
        vector: List[float],
        modelo: str,
        estrategia_chunking: str,
        contenido_chunk: Optional[str] = None
    ) -> int:
        """
        Persiste el embedding vectorial de un chunk de texto en pgvector.

        CAMBIO v2: acepta contenido_chunk opcional. Cuando se provee, se
        guarda junto al vector para que buscar_chunks_similares pueda
        devolver el texto del fragmento sin hacer JOIN con Documento.

        SQL:
            INSERT INTO Embedding_Texto
                (id_doc, chunk_id, vector, modelo, estrategia_chunking, contenido_chunk)
            VALUES ($1, $2, $3::vector, $4, $5, $6)
            RETURNING id_embedding

        Args:
            id_doc: ID del documento al que pertenece el chunk.
            chunk_id: Número de chunk dentro del documento (empieza en 0).
            vector: Embedding numérico generado por el modelo.
            modelo: Nombre del modelo usado.
            estrategia_chunking: Estrategia usada para dividir el texto.
            contenido_chunk: Texto real del fragmento (opcional pero recomendado).

        Returns:
            id_embedding asignado por la base de datos.

        Raises:
            ValueError: Si el vector está vacío, id_doc no es positivo,
                        o chunk_id es negativo.
            RuntimeError: Si hay error en la operación de BD.
        """
        if not isinstance(id_doc, int) or id_doc <= 0:
            raise ValueError("id_doc debe ser un entero positivo")
        if not isinstance(chunk_id, int) or chunk_id < 0:
            raise ValueError("chunk_id debe ser un entero no negativo")
        if not vector:
            raise ValueError("El vector no puede estar vacío")

        vector_str = "[" + ",".join(str(v) for v in vector) + "]"

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                id_embedding = await conexion.fetchval(
                    """
                    INSERT INTO Embedding_Texto
                        (id_doc, chunk_id, vector, modelo, estrategia_chunking, contenido_chunk)
                    VALUES ($1, $2, $3::vector, $4, $5, $6)
                    RETURNING id_embedding
                    """,
                    id_doc,
                    chunk_id,
                    vector_str,
                    modelo,
                    estrategia_chunking,
                    contenido_chunk
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
        Busca los chunks más similares semánticamente a una consulta.

        CAMBIO v2: lee contenido_chunk desde Embedding_Texto en lugar de
        hacer JOIN con Documento para traer contenido_texto completo.
        Esto reduce el payload transferido desde PostgreSQL de potencialmente
        cientos de KB (documento completo × N chunks) a solo los fragmentos
        relevantes (típicamente < 1 KB cada uno).

        El JOIN con Documento se mantiene únicamente para obtener el título,
        que es un campo corto (VARCHAR) necesario para que Claude identifique
        la fuente.

        SQL (sin filtro):
            SELECT
                et.id_doc,
                d.titulo,
                et.chunk_id,
                et.estrategia_chunking      AS estrategia,
                1 - (et.vector <=> $1::vector) AS similitud,
                et.contenido_chunk          AS contenido
            FROM Embedding_Texto et
            JOIN Documento d ON d.id_doc = et.id_doc
            ORDER BY et.vector <=> $1::vector
            LIMIT $2

        Args:
            vector_consulta: Embedding de la consulta del usuario.
            top_k: Número máximo de chunks a retornar.
            estrategia_chunking: Filtro opcional por estrategia de chunking.

        Returns:
            Lista de dicts con las claves:
                - id_doc (int)
                - titulo (str)
                - chunk_id (int)
                - estrategia (str)
                - similitud (float) [0.0, 1.0]
                - contenido (str | None): texto del chunk, no del documento completo

        Raises:
            ValueError: Si vector_consulta está vacío o top_k no es positivo.
            RuntimeError: Si hay error en la operación de BD.
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
                            et.contenido_chunk                  AS contenido
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
                            et.contenido_chunk                  AS contenido
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
                        "id_doc":     fila["id_doc"],
                        "titulo":     fila["titulo"],
                        "chunk_id":   fila["chunk_id"],
                        "estrategia": fila["estrategia"],
                        "similitud":  float(fila["similitud"]),
                        "contenido":  fila["contenido"],
                    }
                    for fila in filas
                ]

        except Exception as e:
            raise RuntimeError(f"Error al buscar chunks similares: {e}") from e

    # ------------------------------------------------------------------
    # OPERACIONES DE EMBEDDINGS DE IMAGEN (pgvector)
    # ------------------------------------------------------------------

    async def guardar_embedding_imagen(
        self,
        id_imagen: int,
        vector: List[float],
        modelo: str
    ) -> int:
        """
        Persiste el embedding vectorial CLIP de una imagen en pgvector.

        SQL:
            INSERT INTO Embedding_Imagen (id_imagen, vector, modelo)
            VALUES ($1, $2::vector, $3)
            RETURNING id_embedding

        Args:
            id_imagen: ID de la imagen a la que pertenece el embedding.
            vector: Embedding CLIP de la imagen (512 dims con CLIP ViT-B/32).
            modelo: Nombre del modelo CLIP usado.

        Returns:
            id_embedding asignado por la base de datos.

        Raises:
            ValueError: Si el vector está vacío o id_imagen no es positivo.
            RuntimeError: Si hay error en la operación de BD.
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

    async def buscar_imagenes_similares(
        self,
        vector_consulta: List[float],
        top_k: int
    ) -> List[dict]:
        """
        Busca imagenes similares usando embeddings CLIP almacenados en pgvector.
        """
        if not vector_consulta:
            raise ValueError("El vector de consulta no puede estar vacio")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k debe ser un entero positivo")

        vector_str = "[" + ",".join(str(v) for v in vector_consulta) + "]"

        try:
            async with conexion_bd.obtener_conexion() as conexion:
                async with conexion.transaction():
                    await conexion.execute("SET LOCAL ivfflat.probes = 100")
                    filas = await conexion.fetch(
                        """
                        SELECT
                            i.id_imagen,
                            i.ruta_archivo,
                            i.descripcion,
                            i.etiquetas,
                            i.id_doc,
                            d.titulo AS titulo_documento,
                            d.fuente AS fuente_documento,
                            o.id_objeto,
                            o.nombre AS nombre_objeto,
                            o.descripcion_cientifica,
                            CASE
                                WHEN g.id_objeto IS NOT NULL THEN 'galaxia'
                                WHEN se.id_objeto IS NOT NULL THEN 'sistema_estelar'
                                WHEN e.id_objeto IS NOT NULL THEN 'estrella'
                                WHEN p.id_objeto IS NOT NULL THEN 'planeta'
                                WHEN l.id_objeto IS NOT NULL THEN 'luna'
                                ELSE NULL
                            END AS tipo_objeto,
                            1 - (ei.vector <=> $1::vector) AS similitud
                        FROM Embedding_Imagen ei
                        JOIN Imagen i ON i.id_imagen = ei.id_imagen
                        LEFT JOIN Documento d ON d.id_doc = i.id_doc
                        LEFT JOIN Objeto_Astronomico o ON o.id_objeto = d.id_objeto
                        LEFT JOIN Galaxia g ON g.id_objeto = o.id_objeto
                        LEFT JOIN Sistema_Estelar se ON se.id_objeto = o.id_objeto
                        LEFT JOIN Estrella e ON e.id_objeto = o.id_objeto
                        LEFT JOIN Planeta p ON p.id_objeto = o.id_objeto
                        LEFT JOIN Luna l ON l.id_objeto = o.id_objeto
                        ORDER BY ei.vector <=> $1::vector
                        LIMIT $2
                        """,
                        vector_str,
                        top_k
                    )
                return [
                    {
                        "id_imagen": fila["id_imagen"],
                        "ruta_archivo": fila["ruta_archivo"],
                        "descripcion": fila["descripcion"],
                        "etiquetas": fila["etiquetas"],
                        "id_doc": fila["id_doc"],
                        "titulo_documento": fila["titulo_documento"],
                        "fuente_documento": fila["fuente_documento"],
                        "id_objeto": fila["id_objeto"],
                        "nombre_objeto": fila["nombre_objeto"],
                        "descripcion_cientifica": fila["descripcion_cientifica"],
                        "tipo_objeto": fila["tipo_objeto"],
                        "similitud": float(fila["similitud"]),
                    }
                    for fila in filas
                ]
        except Exception as e:
            raise RuntimeError(f"Error al buscar imagenes similares: {e}") from e
