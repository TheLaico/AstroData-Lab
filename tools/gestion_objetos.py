"""
Módulo de herramientas MCP para gestión de objetos astronómicos en AstroData Lab.

Expone herramientas MCP para operaciones CRUD completas en objetos astronómicos:
1. Crear nuevos objetos (galaxias, sistemas, estrellas, planetas, lunas)
2. Obtener objetos por ID o nombre
3. Actualizar objetos (y regenerar embeddings si es necesario)
4. Eliminar objetos (en cascada)
5. Listar planetas habitables con filtros

Implementa el patrón de inyección de dependencias: el codificador de embeddings
se proporciona por parámetro, no se instancia dentro. Sigue SRP: orquesta operaciones
CRUD sin implementar lógica de persistencia directa (delegada a repositorios).

Genera y mantiene embeddings de descripciones de objetos automáticamente para
permitir búsquedas semánticas posteriores.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import re
from mcp.types import Tool, TextContent

from database.repositorio_objetos import RepositorioObjetos
from database.repositorio_documentos import RepositorioDocumentos
from embeddings.interfaz_codificador import CodificadorBase
from models.base_objeto_astronomico import ObjetoAstronomico
from models.galaxia_model import Galaxia
from models.sistema_estelar_model import SistemaEstelar
from models.estrella_model import Estrella
from models.planeta_model import Planeta
from models.luna_model import Luna
from models.documento_model import Documento
from models.imagen_model import Imagen


class GestionObjetos:
    """
    Conjunto de herramientas MCP para gestión CRUD de objetos astronómicos.
    
    Orquesta operaciones completas de creación, lectura, actualización y eliminación
    de objetos astronómicos en la jerarquía de tipos (galaxias, sistemas, estrellas,
    planetas, lunas).
    
    Implementa inyección de dependencias: recibe CodificadorBase en __init__
    para permitir cambiar modelos de embeddings sin modificar el código.
    
    Genera automáticamente embeddings para descripciones científicas, permitiendo
    búsquedas semánticas posteriores sin intervención manual.
    
    Sigue Responsabilidad Única: esta clase solo orquesta CRUD y embedding,
    no implementa persistencia (delegada a repositorios).
    
    Atributos:
        _codificador: Codificador de texto inyectado (abstracción de CodificadorBase)
        _repo_objetos: Repositorio para gestionar objetos astronómicos
        _repo_documentos: Repositorio para gestionar embeddings
    """

    # Tipos de objetos válidos soportados
    TIPOS_VALIDOS = {'galaxia', 'sistema_estelar', 'estrella', 'planeta', 'luna'}

    def __init__(self, codificador: CodificadorBase) -> None:
        """
        Inicializa las herramientas de gestión de objetos con sus dependencias.
        
        Args:
            codificador: Implementación de CodificadorBase (ej: CodificadorTexto)
                        Permite cambiar modelo sin modificar esta clase.
        
        Raises:
            TypeError: Si codificador no implementa CodificadorBase
        """
        if not isinstance(codificador, CodificadorBase):
            raise TypeError(
                "codificador debe implementar la interfaz CodificadorBase"
            )

        self._codificador = codificador
        self._repo_objetos = RepositorioObjetos()
        self._repo_documentos = RepositorioDocumentos()

    def _chunificar_texto(
        self,
        texto: str,
        estrategia: str = 'sentence'
    ) -> List[str]:
        """
        Divide un texto en chunks según la estrategia especificada.
        
        Implementa estrategias de chunking para preparar documentos para
        embeddings. Cada chunk genera su propio embedding para búsqueda
        semántica granular.
        
        Args:
            texto: Texto a dividir
            estrategia: 'sentence' (por oraciones), 'paragraph' (por párrafos),
                       'fixed' (tamaño fijo de 500 caracteres)
        
        Returns:
            Lista de chunks de texto
        
        Raises:
            ValueError: Si estrategia no es válida
        """
        if not texto or not texto.strip():
            return []
        
        if estrategia == 'sentence':
            # Dividir por oraciones (punto, exclamación, interrogación)
            # Manteniendo puntuación
            chunks = re.split(r'(?<=[.!?])\s+', texto.strip())
            # Filtrar chunks vacíos
            return [chunk.strip() for chunk in chunks if chunk.strip()]
        
        elif estrategia == 'paragraph':
            # Dividir por párrafos (saltos de línea dobles)
            chunks = texto.strip().split('\n\n')
            return [chunk.strip() for chunk in chunks if chunk.strip()]
        
        elif estrategia == 'fixed':
            # Dividir en chunks de ~500 caracteres, respetando espacios
            chunk_size = 500
            chunks = []
            current = ''
            
            for palabra in texto.split():
                if len(current) + len(palabra) + 1 > chunk_size:
                    if current:
                        chunks.append(current.strip())
                    current = palabra
                else:
                    current += ' ' + palabra if current else palabra
            
            if current:
                chunks.append(current.strip())
            
            return chunks
        
        else:
            raise ValueError(
                f"Estrategia de chunking no válida: {estrategia}. "
                f"Debe ser: 'sentence', 'paragraph', 'fixed'"
            )

    async def crear_documento_con_embeddings(
        self,
        titulo: str,
        contenido_texto: str,
        id_objeto: int,
        idioma: str = 'es',
        fuente: str = 'manual',
        estrategia_chunking: str = 'sentence'
    ) -> Dict[str, Any]:
        """
        Crea un documento con chunking automático y embeddings para cada chunk.
        
        Orquesta el flujo completo:
        1. Crea el documento en BD
        2. Divide el contenido en chunks según estrategia
        3. Genera embedding para cada chunk con el codificador inyectado
        4. Persiste cada embedding vectorial en Embedding_Texto
        5. Retorna resultado con estadísticas de chunking y embedding
        
        Sigue SRP: orquesta sin implementar persistencia (delegada a repositorios).
        
        Args:
            titulo: Título del documento
            contenido_texto: Contenido completo a chunificar
            id_objeto: ID del objeto astronómico relacionado
            idioma: Código de idioma ISO 639-1 (ej: 'es', 'en', 'fr')
            fuente: Origen del documento (ej: 'NASA', 'ESA', 'manual')
            estrategia_chunking: Estrategia a usar ('sentence', 'paragraph', 'fixed')
        
        Returns:
            Dict con estructura:
            {
                'id_doc': int,
                'titulo': str,
                'contenido_texto': str,
                'id_objeto': int,
                'num_chunks': int,
                'chunks_con_embedding': int,
                'embeddings_generados': bool,
                'estrategia_chunking': str,
                'fecha_creacion': str,
                'chunk_ids': [int, ...]  # IDs de embeddings persistidos
            }
            
            O en caso de error:
            {
                'error': str,
                'detalles': str
            }
        
        Example:
            >>> resultado = await gestion.crear_documento_con_embeddings(
            ...     titulo='Características de Marte',
            ...     contenido_texto='Marte es un planeta rocoso. Tiene una atmósfera delgada.'
            ...                      'Su temperatura promedio es de -65°C.',
            ...     id_objeto=5,
            ...     estrategia_chunking='sentence'
            ... )
            >>> resultado['num_chunks']
            3
            >>> resultado['embeddings_generados']
            True
        """
        try:
            # 1. Validar inputs
            if not titulo or not titulo.strip():
                return {'error': 'El título no puede estar vacío', 'detalles': ''}
            
            if not contenido_texto or not contenido_texto.strip():
                return {'error': 'El contenido no puede estar vacío', 'detalles': ''}
            
            if id_objeto <= 0:
                return {'error': 'id_objeto debe ser positivo', 'detalles': ''}
            
            # 2. Crear documento en BD
            documento = Documento(
                id_doc=0,  # Asignado por BD
                titulo=titulo.strip(),
                idioma=idioma,
                fecha=datetime.now(),
                fuente=fuente,
                contenido_texto=contenido_texto.strip(),
                id_objeto=id_objeto
            )
            
            documento_creado = await self._repo_documentos.crear_documento(documento)
            
            # 3. Chunificar contenido
            chunks = self._chunificar_texto(
                contenido_texto.strip(),
                estrategia_chunking
            )
            
            if not chunks:
                return {
                    'error': 'El chunking no generó fragmentos válidos',
                    'detalles': f'Estrategia: {estrategia_chunking}'
                }
            
            # 4 & 5. Generar embeddings para cada chunk
            chunk_ids = []
            chunks_con_embedding = 0
            
            for chunk_id, chunk_contenido in enumerate(chunks):
                try:
                    # Generar embedding del chunk
                    vector = await self._codificador.codificar_texto(chunk_contenido)
                    
                    # Persistir embedding
                    id_embedding = await self._repo_documentos.guardar_embedding_texto(
                        id_doc=documento_creado.id_doc,
                        chunk_id=chunk_id,
                        vector=vector,
                        modelo=self._codificador.nombre_modelo,
                        estrategia_chunking=estrategia_chunking
                    )
                    
                    chunk_ids.append(id_embedding)
                    chunks_con_embedding += 1
                
                except Exception:
                    # Continuar con próximo chunk si éste falla
                    pass
            
            embeddings_exitosos = chunks_con_embedding == len(chunks)
            
            # 6. Retornar respuesta
            return {
                'id_doc': documento_creado.id_doc,
                'titulo': documento_creado.titulo,
                'contenido_texto': documento_creado.contenido_texto[:200] + '...',
                'id_objeto': documento_creado.id_objeto,
                'num_chunks': len(chunks),
                'chunks_con_embedding': chunks_con_embedding,
                'embeddings_generados': embeddings_exitosos,
                'estrategia_chunking': estrategia_chunking,
                'fecha_creacion': datetime.now().isoformat(),
                'chunk_ids': chunk_ids
            }
        
        except Exception as e:
            return {
                'error': 'Error al crear documento con embeddings',
                'detalles': str(e)
            }

    async def crear_imagen_con_embedding(
        self,
        ruta_archivo: str,
        descripcion: str,
        etiquetas: List[str],
        id_doc: int
    ) -> Dict[str, Any]:
        """
        Crea un registro de imagen con embedding vectorial automático.
        
        Orquesta el flujo:
        1. Crea registro de imagen en BD
        2. Genera embedding CLIP de la imagen
        3. Persiste embedding en Embedding_Imagen
        4. Retorna resultado
        
        Requiere que el codificador inyectado sea capaz de procesar imágenes.
        
        Args:
            ruta_archivo: Ruta al archivo de imagen (ej: '/datos/galaxias/m31.jpg')
            descripcion: Descripción textual de la imagen
            etiquetas: Lista de etiquetas para categorización (ej: ['galaxia', 'espiral'])
            id_doc: ID del documento asociado
        
        Returns:
            Dict con estructura:
            {
                'id_imagen': int,
                'ruta_archivo': str,
                'descripcion': str,
                'etiquetas': list,
                'id_doc': int,
                'embedding_generado': bool,
                'fecha_creacion': str
            }
            
            O en caso de error:
            {
                'error': str,
                'detalles': str
            }
        
        Example:
            >>> resultado = await gestion.crear_imagen_con_embedding(
            ...     ruta_archivo='/datos/galaxias/andromeda.jpg',
            ...     descripcion='Galaxia de Andromeda capturada por Hubble',
            ...     etiquetas=['galaxia', 'andromeda', 'espiral'],
            ...     id_doc=3
            ... )
            >>> resultado['embedding_generado']
            True
        """
        try:
            # 1. Validar inputs
            if not ruta_archivo or not ruta_archivo.strip():
                return {'error': 'La ruta del archivo no puede estar vacía', 'detalles': ''}
            
            if not descripcion or not descripcion.strip():
                return {'error': 'La descripción no puede estar vacía', 'detalles': ''}
            
            if id_doc <= 0:
                return {'error': 'id_doc debe ser positivo', 'detalles': ''}
            
            # 2. Crear imagen en BD
            imagen = Imagen(
                id_imagen=0,  # Asignado por BD
                ruta_archivo=ruta_archivo.strip(),
                descripcion=descripcion.strip(),
                etiquetas=etiquetas or [],
                id_doc=id_doc
            )
            
            imagen_creada = await self._repo_documentos.crear_imagen(imagen)
            
            # 3. Generar embedding de imagen
            embedding_generado = False
            try:
                # Usar codificador para procesar imagen
                # El codificador debe tener método codificar_imagen()
                if hasattr(self._codificador, 'codificar_imagen'):
                    vector = await self._codificador.codificar_imagen(ruta_archivo.strip())
                    
                    # 4. Persistir embedding
                    await self._repo_documentos.guardar_embedding_imagen(
                        id_imagen=imagen_creada.id_imagen,
                        vector=vector,
                        modelo=self._codificador.nombre_modelo
                    )
                    
                    embedding_generado = True
                else:
                    # Codificador no soporta imágenes
                    pass
            
            except Exception:
                # Si falla embedding, la imagen ya fue creada
                embedding_generado = False
            
            # 5. Retornar respuesta
            return {
                'id_imagen': imagen_creada.id_imagen,
                'ruta_archivo': imagen_creada.ruta_archivo,
                'descripcion': imagen_creada.descripcion,
                'etiquetas': imagen_creada.etiquetas,
                'id_doc': imagen_creada.id_doc,
                'embedding_generado': embedding_generado,
                'fecha_creacion': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'error': 'Error al crear imagen con embedding',
                'detalles': str(e)
            }

    async def _crear_por_tipo(
        self,
        tipo: str,
        nombre: str,
        descripcion_cientifica: str,
        atributos: dict
    ):
        """
        Crea el objeto astronómico completo delegando directamente al método
        específico del repositorio. Cada método crear_X del repositorio realiza
        su propio INSERT en Objeto_Astronomico y en la tabla del tipo, por lo
        que NO se llama a crear_objeto() por separado.

        Retorna el objeto creado o un dict con 'error' si fallan las validaciones.
        """
        # ── Planeta ────────────────────────────────────────────────────────
        if tipo == "planeta":
            id_sistema = atributos.get("id_sistema")
            if id_sistema is None:
                return {
                    "error": "Falta atributo requerido: id_sistema",
                    "detalles": "Un planeta debe pertenecer a un sistema estelar existente."
                }
            planeta = Planeta(
                id_objeto=0,  # Asignado por la BD al insertar
                nombre=nombre,
                descripcion_cientifica=descripcion_cientifica,
                id_tipo_planeta=atributos.get("id_tipo_planeta", 1),
                id_sistema=int(id_sistema),
                masa=float(atributos.get("masa", atributos.get("masa_masas_terrestres", 1.0))),
                temperatura=int(atributos.get("temperatura", atributos.get("temperatura_K", 288)))
            )
            return await self._repo_objetos.crear_planeta(planeta)

        # ── Estrella ───────────────────────────────────────────────────────
        elif tipo == "estrella":
            id_sistema = atributos.get("id_sistema")
            if id_sistema is None:
                return {
                    "error": "Falta atributo requerido: id_sistema",
                    "detalles": "Una estrella debe pertenecer a un sistema estelar existente."
                }
            estrella = Estrella(
                id_objeto=0,  # Asignado por la BD al insertar
                nombre=nombre,
                descripcion_cientifica=descripcion_cientifica,
                id_tipo_estrella=atributos.get("id_tipo_estrella", 1),
                id_sistema=int(id_sistema),
                masa=float(atributos.get("masa", 1.0)),
                temperatura=float(atributos.get("temperatura", atributos.get("temperatura_K", 5778.0)))
            )
            return await self._repo_objetos.crear_estrella(estrella)

        # ── Sistema Estelar ────────────────────────────────────────────────
        elif tipo == "sistema_estelar":
            id_galaxia = atributos.get("id_galaxia")
            if id_galaxia is None:
                return {
                    "error": "Falta atributo requerido: id_galaxia",
                    "detalles": "Un sistema estelar debe pertenecer a una galaxia existente."
                }
            sistema = SistemaEstelar(
                id_objeto=0,  # Asignado por la BD al insertar
                nombre=nombre,
                descripcion_cientifica=descripcion_cientifica,
                id_galaxia=int(id_galaxia)
            )
            return await self._repo_objetos.crear_sistema_estelar(sistema)

        # ── Galaxia ────────────────────────────────────────────────────────
        elif tipo == "galaxia":
            galaxia = Galaxia(
                id_objeto=0,  # Asignado por la BD al insertar
                nombre=nombre,
                descripcion_cientifica=descripcion_cientifica,
                id_tipo_galaxia=atributos.get("id_tipo_galaxia", 1),
                distancia=float(atributos.get("distancia", 0.0))
            )
            return await self._repo_objetos.crear_galaxia(galaxia)

        # ── Luna ───────────────────────────────────────────────────────────
        elif tipo == "luna":
            id_planeta = atributos.get("id_planeta")
            if id_planeta is None:
                return {
                    "error": "Falta atributo requerido: id_planeta",
                    "detalles": "Una luna debe orbitar un planeta existente."
                }
            luna = Luna(
                id_objeto=0,  # Asignado por la BD al insertar
                nombre=nombre,
                descripcion_cientifica=descripcion_cientifica,
                id_planeta=int(id_planeta),
                radio=float(atributos.get("radio", 0.0))
            )
            return await self._repo_objetos.crear_luna(luna)

        return {
            "error": f"Tipo no soportado: {tipo}",
            "detalles": ""
        }

    async def crear_objeto_astronomico(
        self,
        nombre: str,
        tipo: str,
        descripcion_cientifica: str,
        atributos: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Crea un nuevo objeto astronómico en la base de datos con su embedding.
        
        Proceso:
        1. Valida que tipo sea uno de: galaxia, sistema_estelar, estrella, planeta, luna
        2. Valida que nombre y descripcion_cientifica no estén vacíos
        3. Inserta en la tabla específica del tipo (el repositorio gestiona también
           el INSERT en Objeto_Astronomico)
        4. Vectoriza descripcion_cientifica con el codificador inyectado
        5. Persiste el vector en Embedding_Texto (chunk_id=0, estrategia='descripcion')
        6. Retorna objeto completo con id asignado y estado real del embedding
        
        Args:
            nombre: Nombre del objeto (ej: "Vía Láctea", "Proxima Centauri")
            tipo: Tipo de objeto, uno de: 'galaxia', 'sistema_estelar', 'estrella',
                  'planeta', 'luna'
            descripcion_cientifica: Descripción técnica del objeto para embeddings
            atributos: Dict con atributos específicos del tipo
                      Ej: {'id_tipo_planeta': 1, 'masa_masas_terrestres': 1.0,
                           'temperatura_K': 288, 'id_sistema': 5}
        
        Returns:
            Dict con estructura:
            {
                'id_objeto': int,
                'nombre': str,
                'tipo': str,
                'descripcion_cientifica': str,
                'embedding_generado': bool,  # True solo si se persistió correctamente
                'fecha_creacion': str (ISO format),
                'atributos': dict
            }
            
            O en caso de error:
            {
                'error': str (mensaje en español),
                'detalles': str (excepción técnica)
            }
        
        Example:
            >>> resultado = await gestion.crear_objeto_astronomico(
            ...     nombre="Kepler-452b",
            ...     tipo="planeta",
            ...     descripcion_cientifica="Exoplaneta superterrestre en órbita habitable",
            ...     atributos={
            ...         'id_tipo_planeta': 1,
            ...         'masa_masas_terrestres': 5.0,
            ...         'temperatura_K': 265,
            ...         'id_sistema': 3
            ...     }
            ... )
            >>> resultado['id_objeto']
            42
        """
        try:
            # 1. Validar tipo
            if tipo.lower() not in self.TIPOS_VALIDOS:
                return {
                    'error': f"Tipo de objeto no válido: {tipo}",
                    'detalles': f"Debe ser uno de: {', '.join(self.TIPOS_VALIDOS)}"
                }

            tipo = tipo.lower()

            # 2. Validar inputs básicos
            if not nombre or not nombre.strip():
                return {
                    'error': "El nombre del objeto no puede estar vacío",
                    'detalles': ""
                }

            if not descripcion_cientifica or not descripcion_cientifica.strip():
                return {
                    'error': "La descripción científica no puede estar vacía",
                    'detalles': ""
                }

            # 3. Crear objeto en BD (un solo INSERT por método crear_X del repositorio)
            objeto = await self._crear_por_tipo(
                tipo=tipo,
                nombre=nombre.strip(),
                descripcion_cientifica=descripcion_cientifica.strip(),
                atributos=atributos
            )

            if isinstance(objeto, dict) and 'error' in objeto:
                return objeto

            # 4 & 5. Vectorizar la descripción y persistir el embedding.
            #        embedding_generado refleja el resultado real de la persistencia:
            #        True solo si tanto la codificación como el INSERT en BD tuvieron éxito.
            embedding_generado = False
            try:
                vector = await self._codificador.codificar_texto(
                    descripcion_cientifica.strip()
                )
                await self._repo_documentos.guardar_embedding_texto(
                    id_doc=objeto.id_objeto,
                    chunk_id=0,
                    vector=vector,
                    modelo=self._codificador.nombre_modelo,
                    estrategia_chunking='descripcion'
                )
                embedding_generado = True
            except Exception:
                # El objeto ya fue creado; el embedding fallido no revierte la creación.
                # El llamador puede reintentar la vectorización de forma independiente.
                embedding_generado = False

            # 6. Retornar respuesta
            return {
                'id_objeto': objeto.id_objeto,
                'nombre': objeto.nombre,
                'tipo': tipo,
                'descripcion_cientifica': objeto.descripcion_cientifica,
                'embedding_generado': embedding_generado,
                'fecha_creacion': datetime.now().isoformat(),
                'atributos': atributos
            }

        except Exception as e:
            return {
                'error': "Error al crear objeto astronómico",
                'detalles': str(e)
            }

    async def obtener_objeto_astronomico(
        self,
        id_objeto: Optional[int] = None,
        nombre: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtiene un objeto astronómico por su ID o nombre.
        
        Busca en la base de datos un objeto que coincida con el ID proporcionado
        o el nombre (búsqueda case-insensitive parcial).
        
        Args:
            id_objeto: ID único del objeto (recomendado si se conoce)
            nombre: Nombre del objeto para búsqueda parcial case-insensitive
                   (ej: "terra" coincide con "Tierra")
        
        Returns:
            Dict con estructura:
            {
                'id_objeto': int,
                'nombre': str,
                'descripcion_cientifica': str,
                'encontrado': True
            }
            
            O en caso de error:
            {
                'error': str,
                'encontrado': False
            }
        
        Example:
            >>> resultado = await gestion.obtener_objeto_astronomico(id_objeto=1)
            >>> resultado['nombre']
            'Vía Láctea'
        """
        try:
            # Validar que al menos un parámetro se proporcione
            if id_objeto is None and (nombre is None or not nombre.strip()):
                return {
                    'error': "Debe proporcionar id_objeto o nombre",
                    'encontrado': False
                }

            objeto = None

            # Buscar por ID si se proporciona
            if id_objeto is not None:
                try:
                    objeto = await self._repo_objetos.obtener_objeto_por_id(id_objeto)
                except Exception:
                    pass

            # Buscar por nombre si no se encontró por ID
            if objeto is None and nombre and nombre.strip():
                try:
                    objeto = await self._repo_objetos.obtener_objeto_por_nombre(
                        nombre.strip()
                    )
                except Exception:
                    pass

            if objeto is None:
                return {
                    'error': "Objeto astronómico no encontrado",
                    'encontrado': False
                }

            return {
                'id_objeto': objeto.id_objeto,
                'nombre': objeto.nombre,
                'descripcion_cientifica': objeto.descripcion_cientifica,
                'encontrado': True
            }

        except Exception as e:
            return {
                'error': f"Error al obtener objeto: {str(e)}",
                'encontrado': False
            }

    async def actualizar_objeto_astronomico(
        self,
        id_objeto: int,
        campos: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Actualiza campos de un objeto astronómico existente.
        
        Actualiza los campos indicados en el dict. Si se actualiza
        descripcion_cientifica, regenera automáticamente su embedding
        para mantener la sincronización con búsquedas semánticas.
        
        Args:
            id_objeto: ID del objeto a actualizar
            campos: Dict con pares clave:valor a actualizar
                   Soporta: 'descripcion_cientifica' (regenera embedding),
                   y otros campos según tipo de objeto
        
        Returns:
            Dict con estructura:
            {
                'id_objeto': int,
                'nombre': str,
                'descripcion_cientifica': str,
                'actualizado': True,
                'embedding_regenerado': bool,
                'fecha_actualizacion': str
            }
            
            O en caso de error:
            {
                'error': str,
                'detalles': str,
                'actualizado': False
            }
        
        Example:
            >>> resultado = await gestion.actualizar_objeto_astronomico(
            ...     id_objeto=5,
            ...     campos={'descripcion_cientifica': 'Nueva descripción'}
            ... )
            >>> resultado['embedding_regenerado']
            True
        """
        try:
            # Validar que el objeto exista
            objeto = await self._repo_objetos.obtener_objeto_por_id(id_objeto)
            if objeto is None:
                return {
                    'error': "Objeto no encontrado",
                    'detalles': f"ID: {id_objeto}",
                    'actualizado': False
                }

            embedding_regenerado = False

            # Si se actualiza descripción, actualizar en BD y regenerar embedding
            if 'descripcion_cientifica' in campos:
                nueva_descripcion = campos['descripcion_cientifica']

                if isinstance(nueva_descripcion, str) and nueva_descripcion.strip():
                    # Actualizar descripción en BD
                    objeto = await self._repo_objetos.actualizar_descripcion(
                        id_objeto=id_objeto,
                        nueva_descripcion=nueva_descripcion.strip()
                    )

                    # Regenerar y persistir embedding
                    try:
                        vector = await self._codificador.codificar_texto(
                            nueva_descripcion.strip()
                        )
                        await self._repo_documentos.guardar_embedding_texto(
                            id_doc=objeto.id_objeto,
                            chunk_id=0,
                            vector=vector,
                            modelo=self._codificador.nombre_modelo,
                            estrategia_chunking='descripcion'
                        )
                        embedding_regenerado = True
                    except Exception:
                        embedding_regenerado = False
                else:
                    return {
                        'error': "Descripción científica vacía",
                        'detalles': "",
                        'actualizado': False
                    }

            return {
                'id_objeto': objeto.id_objeto,
                'nombre': objeto.nombre,
                'descripcion_cientifica': objeto.descripcion_cientifica,
                'actualizado': True,
                'embedding_regenerado': embedding_regenerado,
                'fecha_actualizacion': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'error': "Error al actualizar objeto",
                'detalles': str(e),
                'actualizado': False
            }

    async def eliminar_objeto_astronomico(self, id_objeto: int) -> Dict[str, Any]:
        """
        Elimina un objeto astronómico de la base de datos.
        
        Realiza eliminación en cascada: elimina el objeto y todas sus
        referencias asociadas (embeddings, documentos, características, etc.)
        manteniendo la integridad referencial de la BD.
        
        Args:
            id_objeto: ID del objeto a eliminar
        
        Returns:
            Dict con estructura:
            {
                'eliminado': True,
                'id_objeto': int,
                'nombre': str,
                'fecha_eliminacion': str,
                'confirmacion': str
            }
            
            O en caso de error:
            {
                'error': str,
                'detalles': str,
                'eliminado': False
            }
        
        Example:
            >>> resultado = await gestion.eliminar_objeto_astronomico(id_objeto=42)
            >>> resultado['eliminado']
            True
        """
        try:
            # Obtener datos del objeto para confirmación
            objeto = await self._repo_objetos.obtener_objeto_por_id(id_objeto)
            if objeto is None:
                return {
                    'error': "Objeto no encontrado",
                    'detalles': f"ID: {id_objeto}",
                    'eliminado': False
                }

            # Realizar eliminación
            eliminado = await self._repo_objetos.eliminar_objeto(id_objeto)

            if not eliminado:
                return {
                    'error': "No se pudo eliminar el objeto",
                    'detalles': f"El repositorio retornó False para ID: {id_objeto}",
                    'eliminado': False
                }

            return {
                'eliminado': True,
                'id_objeto': id_objeto,
                'nombre': objeto.nombre,
                'fecha_eliminacion': datetime.now().isoformat(),
                'confirmacion': f"Objeto '{objeto.nombre}' eliminado completamente con cascada"
            }

        except Exception as e:
            return {
                'error': "Error al eliminar objeto",
                'detalles': str(e),
                'eliminado': False
            }

    async def listar_planetas_habitables(
        self,
        puntaje_minimo: float = 0.5,
        caracteristicas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Lista planetas habitables filtrados por puntaje de habitabilidad.
        
        Recupera planetas cuyo puntaje de habitabilidad sea >= puntaje_minimo,
        opcionalmente filtrando por características ambientales específicas
        (composición, temperatura habitable, agua, etc.).
        
        Args:
            puntaje_minimo: Puntaje mínimo de habitabilidad (0.0 - 1.0, default: 0.5)
            caracteristicas: Lista opcional de características para filtrar
                            (ej: ['agua_liquida', 'atmosfera_oxigeno', 'temperatura_moderada'])
        
        Returns:
            Dict con estructura:
            {
                'planetas': [
                    {
                        'id_objeto': int,
                        'nombre': str,
                        'descripcion_cientifica': str,
                        'masa': float,
                        'temperatura': int,
                        'puntaje_habitabilidad': float
                    },
                    ...
                ],
                'total': int,
                'puntaje_minimo': float,
                'caracteristicas_filtro': list
            }
            
            O en caso de error:
            {
                'error': str,
                'detalles': str,
                'planetas': [],
                'total': 0
            }
        
        Example:
            >>> resultado = await gestion.listar_planetas_habitables(
            ...     puntaje_minimo=0.7,
            ...     caracteristicas=['agua_liquida']
            ... )
            >>> resultado['total']
            3
        """
        try:
            # Validar puntaje_minimo
            if not (0.0 <= puntaje_minimo <= 1.0):
                return {
                    'error': "puntaje_minimo debe estar entre 0.0 y 1.0",
                    'detalles': f"Valor recibido: {puntaje_minimo}",
                    'planetas': [],
                    'total': 0
                }

            # Obtener planetas habitables
            planetas = await self._repo_objetos.listar_planetas_por_habitabilidad(
                puntaje_minimo=puntaje_minimo
            )

            # Convertir a dicts para respuesta
            planetas_lista = [
                {
                    'id_objeto': planeta.id_objeto,
                    'nombre': planeta.nombre,
                    'descripcion_cientifica': planeta.descripcion_cientifica,
                    'masa': planeta.masa,
                    'temperatura': planeta.temperatura,
                    'puntaje_habitabilidad': puntaje_minimo
                }
                for planeta in planetas
            ]

            # TODO: Filtrar por características cuando se implemente en repositorio

            return {
                'planetas': planetas_lista,
                'total': len(planetas_lista),
                'puntaje_minimo': puntaje_minimo,
                'caracteristicas_filtro': caracteristicas or []
            }

        except Exception as e:
            return {
                'error': "Error al listar planetas habitables",
                'detalles': str(e),
                'planetas': [],
                'total': 0
            }

    def obtener_definiciones_tools(self) -> List[Tool]:
        """
        Retorna las definiciones de herramientas MCP para registro en el servidor.
        
        Genera Tool objects que describen a Claude la interfaz de cada herramienta
        CRUD, incluyendo nombres, descripciones, y esquemas de entrada JSON.
        
        Returns:
            List[Tool] con 5 definiciones de herramientas MCP
        
        Estructura de cada Tool:
            - name: Identificador de herramienta para Claude
            - description: Descripción en español de qué hace
            - inputSchema: JSON Schema describiendo los parámetros
        """
        return [
            Tool(
                name="crear_objeto_astronomico",
                description=(
                    "Crea un nuevo objeto astronómico en la base de datos con su descripción "
                    "científica. Genera y persiste automáticamente el embedding de la descripción "
                    "para búsquedas semánticas posteriores."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "nombre": {
                            "type": "string",
                            "description": "Nombre del objeto (ej: 'Vía Láctea', 'Kepler-452b')"
                        },
                        "tipo": {
                            "type": "string",
                            "enum": ["galaxia", "sistema_estelar", "estrella", "planeta", "luna"],
                            "description": "Tipo de objeto astronómico"
                        },
                        "descripcion_cientifica": {
                            "type": "string",
                            "description": "Descripción técnica para embeddings semánticos"
                        },
                        "atributos": {
                            "type": "object",
                            "description": (
                                "Atributos específicos del tipo "
                                "(masa en masas terrestres, temperatura en Kelvin, id_sistema, etc.)"
                            )
                        }
                    },
                    "required": ["nombre", "tipo", "descripcion_cientifica", "atributos"]
                }
            ),
            Tool(
                name="obtener_objeto_astronomico",
                description="Obtiene un objeto astronómico por su ID o nombre.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_objeto": {
                            "type": "integer",
                            "description": "ID único del objeto"
                        },
                        "nombre": {
                            "type": "string",
                            "description": "Nombre del objeto para búsqueda (case-insensitive)"
                        }
                    }
                }
            ),
            Tool(
                name="actualizar_objeto_astronomico",
                description=(
                    "Actualiza campos de un objeto astronómico. Si se modifica "
                    "descripcion_cientifica, regenera y persiste su embedding automáticamente."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_objeto": {
                            "type": "integer",
                            "description": "ID del objeto a actualizar"
                        },
                        "campos": {
                            "type": "object",
                            "description": "Pares clave:valor con campos a actualizar"
                        }
                    },
                    "required": ["id_objeto", "campos"]
                }
            ),
            Tool(
                name="eliminar_objeto_astronomico",
                description=(
                    "Elimina un objeto astronómico de la base de datos "
                    "con eliminación en cascada de referencias."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_objeto": {
                            "type": "integer",
                            "description": "ID del objeto a eliminar"
                        }
                    },
                    "required": ["id_objeto"]
                }
            ),
            Tool(
                name="listar_planetas_habitables",
                description="Lista planetas filtrados por puntaje mínimo de habitabilidad.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "puntaje_minimo": {
                            "type": "number",
                            "description": "Puntaje mínimo de habitabilidad (0.0-1.0, default: 0.5)"
                        },
                        "caracteristicas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Características ambientales para filtrar (opcional)"
                        }
                    }
                }
            ),
            Tool(
                name="crear_documento_con_embeddings",
                description=(
                    "Crea un documento científico con chunking automático y embeddings "
                    "vectoriales para cada fragmento. Permite búsqueda semántica granular."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "titulo": {
                            "type": "string",
                            "description": "Título del documento"
                        },
                        "contenido_texto": {
                            "type": "string",
                            "description": "Contenido completo a chunificar y vectorizar"
                        },
                        "id_objeto": {
                            "type": "integer",
                            "description": "ID del objeto astronómico relacionado"
                        },
                        "idioma": {
                            "type": "string",
                            "description": "Código ISO 639-1 del idioma (ej: 'es', 'en'). Default: 'es'"
                        },
                        "fuente": {
                            "type": "string",
                            "description": "Origen del documento (ej: 'NASA', 'ESA', 'manual'). Default: 'manual'"
                        },
                        "estrategia_chunking": {
                            "type": "string",
                            "enum": ["sentence", "paragraph", "fixed"],
                            "description": (
                                "Estrategia de división: "
                                "'sentence' (por oraciones, recomendado), "
                                "'paragraph' (por párrafos), "
                                "'fixed' (bloques de ~500 caracteres)"
                            )
                        }
                    },
                    "required": ["titulo", "contenido_texto", "id_objeto"]
                }
            ),
            Tool(
                name="crear_imagen_con_embedding",
                description=(
                    "Crea un registro de imagen astronómica con embedding vectorial CLIP "
                    "automático para búsqueda visual semántica."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ruta_archivo": {
                            "type": "string",
                            "description": "Ruta al archivo de imagen (ej: '/datos/galaxias/m31.jpg')"
                        },
                        "descripcion": {
                            "type": "string",
                            "description": "Descripción textual de la imagen"
                        },
                        "etiquetas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Etiquetas para categorización (ej: ['galaxia', 'espiral'])"
                        },
                        "id_doc": {
                            "type": "integer",
                            "description": "ID del documento científico asociado"
                        }
                    },
                    "required": ["ruta_archivo", "descripcion", "id_doc"]
                }
            )
        ]