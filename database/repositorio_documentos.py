"""
Módulo de repositorio para documentos e imágenes en AstroData Lab.

Proporciona la capa de acceso a datos para documentos científicos, imágenes astronómicas
y sus embeddings vectoriales. Gestiona tanto la capa relacional (tablas Documento, Imagen)
como la capa vectorial (pgvector con Embedding_Texto e Embedding_Imagen).
"""

from typing import List, Optional
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