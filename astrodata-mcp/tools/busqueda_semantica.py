"""
Módulo de herramientas MCP para búsqueda semántica en AstroData Lab.

Expone herramientas MCP para búsquedas vectoriales avanzadas:
1. Buscar documentos científicos por similitud semántica
2. Buscar imágenes astronómicas usando queries en texto (cross-modal)
3. Encontrar planetas similares a uno de referencia

Implementa el patrón de inyección de dependencias: el codificador de embeddings
se proporciona por parámetro, no se instancia dentro. Sigue SRP: solo búsqueda
semántica, no persistencia ni transformación de datos (delegada a repositorios).

Utiliza pgvector con operador <=> (distancia L2) para similitud eficiente
en base de datos PostgreSQL.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from mcp.types import Tool, TextContent

from database.repositorio_documentos import RepositorioDocumentos
from database.repositorio_objetos import RepositorioObjetos
from embeddings.interfaz_codificador import CodificadorBase
from models.objeto_astronomico import Planeta


class BusquedaSematica:
    """
    Conjunto de herramientas MCP para búsquedas semánticas vectoriales.
    
    Orquesta búsquedas vectoriales de texto e imágenes usando embeddings,
    permitiendo queries en lenguaje natural para encontrar documentos,
    imágenes y objetos similares.
    
    Implementa inyección de dependencias: recibe CodificadorBase en __init__
    para permitir cambiar modelos de embeddings sin modificar el código.
    
    Sigue Responsabilidad Única: esta clase solo orquesta búsquedas semánticas,
    no implementa persistencia (delegada a repositorios) ni cálculos complejos.
    
    Atributos:
        _codificador: Codificador de texto inyectado (abstracción de CodificadorBase)
        _repo_documentos: Repositorio para gestionar documentos y embeddings
        _repo_objetos: Repositorio para gestionar objetos astronómicos
    """
    
    def __init__(self, codificador: CodificadorBase) -> None:
        """
        Inicializa las herramientas de búsqueda semántica con sus dependencias.
        
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
        self._repo_documentos = RepositorioDocumentos()
        self._repo_objetos = RepositorioObjetos()
    
    
    async def buscar_documentos_semanticos(
        self,
        consulta: str,
        top_k: int = 5,
        estrategia_chunking: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Busca documentos científicos por similitud semántica.
        
        Pipeline:
        1. Vectoriza la consulta en lenguaje natural (384 dimensiones)
        2. Busca los chunks de texto más similares en BD usando pgvector
        3. Retorna documentos ordenados por similitud descendente
        
        Opcionalmente filtra por estrategia de chunking para refinar búsquedas
        (fixed = tamaño fijo, sentence = límites de oraciones, semantic = chunks semánticos).
        
        Args:
            consulta: Pregunta en lenguaje natural
                     (ej: "¿Cómo se forma una galaxia espiral?")
            top_k: Número máximo de documentos/chunks a retornar (default: 5)
            estrategia_chunking: Filtrar por estrategia ('fixed', 'sentence', 'semantic')
                                Si None, busca en todas las estrategias
        
        Returns:
            Dict con estructura:
            {
                'documentos': [
                    {
                        'titulo': str,
                        'chunk_id': int,
                        'estrategia': str,
                        'puntuacion_similitud': float (0.0-1.0)
                    },
                    ...
                ],
                'total': int,
                'consulta': str,
                'estrategia_filtro': str | None
            }
            
            O en caso de error:
            {
                'error': str,
                'detalles': str,
                'documentos': [],
                'total': 0
            }
        
        Example:
            >>> resultado = await busqueda.buscar_documentos_semanticos(
            ...     consulta="Exoplanetas habitables",
            ...     top_k=3,
            ...     estrategia_chunking='semantic'
            ... )
            >>> resultado['documentos'][0]['puntuacion_similitud']
            0.89
        """
        try:
            # Validar inputs
            if not consulta or not consulta.strip():
                return {
                    'error': "La consulta no puede estar vacía",
                    'detalles': "",
                    'documentos': [],
                    'total': 0
                }
            
            if top_k <= 0:
                return {
                    'error': "top_k debe ser mayor a 0",
                    'detalles': f"Valor recibido: {top_k}",
                    'documentos': [],
                    'total': 0
                }
            
            if estrategia_chunking and estrategia_chunking not in ('fixed', 'sentence', 'semantic'):
                return {
                    'error': "estrategia_chunking inválida",
                    'detalles': f"Debe ser 'fixed', 'sentence', 'semantic' u omitirse",
                    'documentos': [],
                    'total': 0
                }
            
            # Vectorizar consulta
            try:
                vector_consulta = await self._codificador.codificar_texto(
                    consulta.strip()
                )
            except Exception as e:
                return {
                    'error': "Error al vectorizar consulta",
                    'detalles': str(e),
                    'documentos': [],
                    'total': 0
                }
            
            # Buscar chunks similares
            try:
                chunks = await self._repo_documentos.buscar_chunks_similares(
                    vector_consulta=vector_consulta,
                    top_k=top_k,
                    estrategia=estrategia_chunking
                )
            except Exception as e:
                return {
                    'error': "Error al buscar chunks similares",
                    'detalles': str(e),
                    'documentos': [],
                    'total': 0
                }
            
            # Formatear resultados
            documentos_lista = []
            for chunk in chunks:
                documentos_lista.append({
                    'titulo': chunk['titulo'],
                    'chunk_id': chunk['chunk_id'],
                    'estrategia': chunk['estrategia_chunking'],
                    'puntuacion_similitud': round(chunk['similitud'], 4)
                })
            
            return {
                'documentos': documentos_lista,
                'total': len(documentos_lista),
                'consulta': consulta.strip(),
                'estrategia_filtro': estrategia_chunking
            }
        
        except Exception as e:
            return {
                'error': "Error al buscar documentos semánticamente",
                'detalles': str(e),
                'documentos': [],
                'total': 0
            }
    
    
    async def buscar_imagenes_semanticas(
        self,
        descripcion: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Busca imágenes astronómicas por descripción en lenguaje natural.
        
        Búsqueda cross-modal: vectoriza la descripción textual en el espacio CLIP
        y busca las imágenes más similares. Permite usar texto para buscar en
        imágenes sin necesidad de proporcionar una imagen como entrada.
        
        Pipeline:
        1. Vectoriza descripción en lenguaje natural (384d con CodificadorTexto)
        2. Busca en embeddings CLIP de imágenes (512d) usando similitud
        3. Retorna imágenes ordenadas por similitud descendente
        
        Args:
            descripcion: Descripción textual de la imagen buscada
                        (ej: "Galaxia espiral azul con anillo de polvo")
            top_k: Número máximo de imágenes a retornar (default: 5)
        
        Returns:
            Dict con estructura:
            {
                'imagenes': [
                    {
                        'ruta_archivo': str,
                        'descripcion': str,
                        'puntuacion_similitud': float (0.0-1.0)
                    },
                    ...
                ],
                'total': int,
                'descripcion_busqueda': str
            }
            
            O en caso de error:
            {
                'error': str,
                'detalles': str,
                'imagenes': [],
                'total': 0
            }
        
        Example:
            >>> resultado = await busqueda.buscar_imagenes_semanticas(
            ...     descripcion="Nebulosa roja con estrellas brillantes",
            ...     top_k=5
            ... )
            >>> resultado['imagenes'][0]['puntuacion_similitud']
            0.85
        """
        try:
            # Validar inputs
            if not descripcion or not descripcion.strip():
                return {
                    'error': "La descripción no puede estar vacía",
                    'detalles': "",
                    'imagenes': [],
                    'total': 0
                }
            
            if top_k <= 0:
                return {
                    'error': "top_k debe ser mayor a 0",
                    'detalles': f"Valor recibido: {top_k}",
                    'imagenes': [],
                    'total': 0
                }
            
            # Vectorizar descripción con codificador de texto
            # (búsqueda cross-modal: texto → imágenes)
            try:
                vector_consulta = await self._codificador.codificar_texto(
                    descripcion.strip()
                )
            except Exception as e:
                return {
                    'error': "Error al vectorizar descripción",
                    'detalles': str(e),
                    'imagenes': [],
                    'total': 0
                }
            
            # Buscar imágenes similares
            try:
                imagenes = await self._repo_documentos.buscar_imagenes_similares(
                    vector_consulta=vector_consulta,
                    top_k=top_k
                )
            except Exception as e:
                return {
                    'error': "Error al buscar imágenes similares",
                    'detalles': str(e),
                    'imagenes': [],
                    'total': 0
                }
            
            # Formatear resultados
            imagenes_lista = []
            for imagen in imagenes:
                imagenes_lista.append({
                    'ruta_archivo': imagen['ruta_archivo'],
                    'descripcion': imagen.get('descripcion', ''),
                    'puntuacion_similitud': round(imagen['similitud'], 4)
                })
            
            return {
                'imagenes': imagenes_lista,
                'total': len(imagenes_lista),
                'descripcion_busqueda': descripcion.strip()
            }
        
        except Exception as e:
            return {
                'error': "Error al buscar imágenes semánticamente",
                'detalles': str(e),
                'imagenes': [],
                'total': 0
            }
    
    
    async def encontrar_planetas_similares(
        self,
        id_planeta: int,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Encuentra planetas similares a uno de referencia en el espacio vectorial.
        
        Pipeline:
        1. Recupera el planeta de referencia desde BD
        2. Obtiene su embedding textual existente, o lo genera desde descripción
        3. Busca otros planetas con embeddings cercanos en el espacio 384d
        4. Excluye el planeta de referencia de los resultados
        5. Retorna lista de planetas similares con scores
        
        Útil para descubrir exoplanetas con características análogas a
        terrestres o buscar sistemas estelares comparables.
        
        Args:
            id_planeta: ID del planeta de referencia
            top_k: Número máximo de planetas similares a retornar (default: 5)
        
        Returns:
            Dict con estructura:
            {
                'planeta_referencia': {
                    'id_objeto': int,
                    'nombre': str,
                    'puntaje_habitabilidad': float
                },
                'planetas_similares': [
                    {
                        'id_objeto': int,
                        'nombre': str,
                        'puntaje_habitabilidad': float,
                        'puntuacion_similitud': float (0.0-1.0)
                    },
                    ...
                ],
                'total': int
            }
            
            O en caso de error:
            {
                'error': str,
                'detalles': str,
                'planeta_referencia': None,
                'planetas_similares': [],
                'total': 0
            }
        
        Example:
            >>> resultado = await busqueda.encontrar_planetas_similares(
            ...     id_planeta=7,
            ...     top_k=3
            ... )
            >>> resultado['planetas_similares'][0]['nombre']
            'Kepler-452b'
        """
        try:
            # Validar inputs
            if id_planeta <= 0:
                return {
                    'error': "id_planeta debe ser positivo",
                    'detalles': f"Valor recibido: {id_planeta}",
                    'planeta_referencia': None,
                    'planetas_similares': [],
                    'total': 0
                }
            
            if top_k <= 0:
                return {
                    'error': "top_k debe ser mayor a 0",
                    'detalles': f"Valor recibido: {top_k}",
                    'planeta_referencia': None,
                    'planetas_similares': [],
                    'total': 0
                }
            
            # Obtener planeta de referencia
            try:
                planeta_ref = await self._repo_objetos.obtener_objeto_por_id(id_planeta)
                if not planeta_ref:
                    return {
                        'error': "Planeta no encontrado",
                        'detalles': f"ID: {id_planeta}",
                        'planeta_referencia': None,
                        'planetas_similares': [],
                        'total': 0
                    }
            except Exception as e:
                return {
                    'error': "Error al obtener planeta de referencia",
                    'detalles': str(e),
                    'planeta_referencia': None,
                    'planetas_similares': [],
                    'total': 0
                }
            
            # Obtener embedding del planeta
            # TODO: Implementar recuperación de embedding existente
            # Por ahora, generamos desde descripción
            try:
                vector_planeta = await self._codificador.codificar_texto(
                    planeta_ref.descripcion_cientifica or planeta_ref.nombre
                )
            except Exception as e:
                return {
                    'error': "Error al vectorizar descripción del planeta",
                    'detalles': str(e),
                    'planeta_referencia': None,
                    'planetas_similares': [],
                    'total': 0
                }
            
            # Buscar planetas similares
            try:
                # Buscar usando chunks_similares pero filtrados para objetos de tipo planeta
                # Por ahora usamos buscar_chunks_similares como proxy
                chunks_similares = await self._repo_documentos.buscar_chunks_similares(
                    vector_consulta=vector_planeta,
                    top_k=top_k + 1  # +1 para excluir el planeta de referencia
                )
            except Exception as e:
                return {
                    'error': "Error al buscar planetas similares",
                    'detalles': str(e),
                    'planeta_referencia': None,
                    'planetas_similares': [],
                    'total': 0
                }
            
            # Formatear resultados
            # TODO: Mejorar para obtener info real de planetas (no solo chunks)
            # Por ahora retornamos estructura base
            planetas_lista = []
            for chunk in chunks_similares:
                # Excluir si es el mismo planeta
                if chunk['titulo'] != planeta_ref.nombre:
                    planetas_lista.append({
                        'nombre': chunk['titulo'],
                        'puntuacion_similitud': round(chunk['similitud'], 4)
                    })
            
            # Limitar a top_k
            planetas_lista = planetas_lista[:top_k]
            
            return {
                'planeta_referencia': {
                    'id_objeto': planeta_ref.id_objeto,
                    'nombre': planeta_ref.nombre
                },
                'planetas_similares': planetas_lista,
                'total': len(planetas_lista)
            }
        
        except Exception as e:
            return {
                'error': "Error al encontrar planetas similares",
                'detalles': str(e),
                'planeta_referencia': None,
                'planetas_similares': [],
                'total': 0
            }
    
    
    def obtener_definiciones_tools(self) -> List[Tool]:
        """
        Retorna las definiciones de herramientas MCP para registro en el servidor.
        
        Genera Tool objects que describen a Claude la interfaz de cada herramienta
        de búsqueda semántica, incluyendo nombres, descripciones, y esquemas
        de entrada JSON.
        
        Returns:
            List[Tool] con 3 definiciones de herramientas MCP
        
        Estructura de cada Tool:
            - name: Identificador de herramienta para Claude
            - description: Descripción en español de qué hace
            - inputSchema: JSON Schema describiendo los parámetros
        """
        return [
            Tool(
                name="buscar_documentos_semanticos",
                description="Busca documentos científicos por similitud semántica usando embeddings. Retorna chunks de texto ordenados por relevancia.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "consulta": {
                            "type": "string",
                            "description": "Pregunta en lenguaje natural (ej: '¿Cómo se forman los agujeros negros?')"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Número máximo de documentos a retornar (default: 5)"
                        },
                        "estrategia_chunking": {
                            "type": "string",
                            "enum": ["fixed", "sentence", "semantic"],
                            "description": "Filtrar por estrategia de chunking (opcional)"
                        }
                    },
                    "required": ["consulta"]
                }
            ),
            Tool(
                name="buscar_imagenes_semanticas",
                description="Busca imágenes astronómicas usando descripción en texto (búsqueda cross-modal texto→imagen).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "descripcion": {
                            "type": "string",
                            "description": "Descripción textual de la imagen buscada (ej: 'Galaxia espiral azul')"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Número máximo de imágenes a retornar (default: 5)"
                        }
                    },
                    "required": ["descripcion"]
                }
            ),
            Tool(
                name="encontrar_planetas_similares",
                description="Encuentra planetas similares a uno de referencia en el espacio vectorial de embeddings.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_planeta": {
                            "type": "integer",
                            "description": "ID del planeta de referencia"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Número máximo de planetas similares a retornar (default: 5)"
                        }
                    },
                    "required": ["id_planeta"]
                }
            )
        ]
