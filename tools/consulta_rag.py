"""
Módulo de herramientas MCP para consultas RAG en AstroData Lab.

Expone dos herramientas MCP que permiten a Claude:
1. Realizar consultas RAG en lenguaje natural (rag_query)
2. Obtener contexto científico de objetos astronómicos específicos (obtener_contexto_objeto)

Implementa el patrón de inyección de dependencias: el codificador de embeddings
se proporciona por parámetro, no se instancia dentro. Sigue SRP: orquesta el flujo RAG
sin implementar lógica de búsqueda o persistencia directa.
"""

from typing import Optional, Dict, Any, List
from mcp.types import Tool, TextContent

from database.repositorio_consultas import RepositorioConsultas
from database.repositorio_documentos import RepositorioDocumentos
from database.repositorio_objetos import RepositorioObjetos
from embeddings.interfaz_codificador import CodificadorBase
from models.consulta import ConsultaEntrada


class ToolsConsultaRAG:
    """
    Conjunto de herramientas MCP para consultas RAG en AstroData Lab.
    
    Orquesta el pipeline completo de Retrieval-Augmented Generation:
    1. Recibe preguntas en lenguaje natural
    2. Vectoriza usando embeddings de texto
    3. Busca similitud semántica en la BD
    4. Recupera documentos relevantes
    5. Retorna contexto estructurado para Claude
    
    Implementa inyección de dependencias: recibe CodificadorBase en __init__
    para permitir cambiar modelos de embeddings sin modificar el código.
    
    Sigue Responsabilidad Única: esta clase solo orquesta, no implementa
    búsqueda vectorial ni persistencia (delegadas a repositorios).
    
    Atributos:
        _codificador: Codificador de texto inyectado (abstracción de CodificadorBase)
        _repo_consultas: Repositorio para gestionar consultas y resultados
        _repo_documentos: Repositorio para gestionar documentos y embeddings
        _repo_objetos: Repositorio para gestionar objetos astronómicos
    """

    def __init__(self, codificador: CodificadorBase) -> None:
        """
        Inicializa las herramientas RAG con sus dependencias.
        
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
        self._repo_consultas = RepositorioConsultas()
        self._repo_documentos = RepositorioDocumentos()
        self._repo_objetos = RepositorioObjetos()

    async def rag_query(
        self,
        texto_pregunta: str,
        top_k: int = 5,
        estrategia_chunking: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Realiza una consulta RAG recuperando documentos similares de la BD.
        
        Pipeline:
        1. Registra la consulta en BD (tabla Consulta)
        2. Vectoriza el texto con CodificadorBase → embedding numérico
        3. Guarda embedding en BD (tabla Embedding_Consulta)
        4. Busca similitud coseno contra Embedding_Texto con buscar_chunks_similares
        5. Construye contexto_para_claude concatenando el contenido de los chunks
        6. Retorna chunks recuperados y contexto listo para el prompt
        
        Args:
            texto_pregunta: Pregunta del usuario en lenguaje natural
                           (ej: "¿Qué planetas son similares a la Tierra?")
            top_k: Número máximo de chunks a recuperar (default: 5)
            estrategia_chunking: Filtrar por estrategia ('fixed', 'sentence', 'semantic')
                                Si None, busca en todas las estrategias
        
        Returns:
            dict con estructura:
            {
                'id_consulta': int,
                'texto_pregunta': str,
                'chunks_recuperados': [
                    {
                        'id_doc': int,
                        'titulo': str,
                        'chunk_id': int,
                        'estrategia': str,
                        'similitud': float,  # 0.0-1.0
                        'contenido': str | None
                    },
                    ...
                ],
                'contexto_para_claude': str,  # Texto concatenado de los chunks
                'fecha_consulta': str         # ISO 8601
            }
            
            En caso de error:
            {
                'error': str,    # Mensaje descriptivo en español
                'detalles': str  # Información técnica adicional
            }
        
        Raises:
            (No lanza excepciones, retorna dict con 'error' en su lugar)
        
        Example:
            >>> tools = ToolsConsultaRAG(codificador=CodificadorTexto())
            >>> resultado = await tools.rag_query(
            ...     texto_pregunta="¿Cuáles son las características de Marte?",
            ...     top_k=3,
            ...     estrategia_chunking='sentence'
            ... )
            >>> resultado['contexto_para_claude']
            "### Observaciones de Marte (chunk 0)\nMarte es..."
        """
        try:
            # --- Validación de entrada ---
            if not texto_pregunta or not texto_pregunta.strip():
                return {
                    'error': 'La pregunta no puede estar vacía',
                    'detalles': 'Se requiere un texto_pregunta válido'
                }

            if top_k <= 0:
                return {
                    'error': 'top_k debe ser positivo',
                    'detalles': f'Recibido: {top_k}'
                }

            # TODO: Obtener id_usuario desde contexto de Claude (por ahora usar 1 temporal)
            id_usuario_temporal = 1

            # 1. Registrar consulta en BD
            entrada_consulta = ConsultaEntrada(
                texto_pregunta=texto_pregunta.strip(),
                id_usuario=id_usuario_temporal
            )
            consulta = await self._repo_consultas.registrar_consulta(entrada_consulta)

            # 2. Vectorizar la pregunta
            vector_embedding = await self._codificador.codificar_texto(
                texto_pregunta.strip()
            )

            # 3. Guardar embedding de la consulta en BD
            await self._repo_consultas.guardar_embedding_consulta(
                id_consulta=consulta.id_consulta,
                vector=vector_embedding,
                modelo=await self._codificador.nombre_modelo()
            )

            # 4. Buscar chunks similares por similitud coseno
            chunks = await self._repo_documentos.buscar_chunks_similares(
                vector_consulta=vector_embedding,
                top_k=top_k,
                estrategia_chunking=estrategia_chunking
            )

            # 5. Construir contexto para Claude concatenando el contenido de los chunks.
            #    Cada chunk se encabeza con el título del documento y su índice para que
            #    Claude pueda citar la fuente con precisión.
            partes_contexto = []
            for chunk in chunks:
                encabezado = f"### {chunk['titulo']} (chunk {chunk['chunk_id']})"
                contenido = chunk.get('contenido') or ''
                partes_contexto.append(f"{encabezado}\n{contenido}".strip())

            contexto_para_claude = "\n\n".join(partes_contexto)

            # 6. Retornar resultado
            return {
                'id_consulta': consulta.id_consulta,
                'texto_pregunta': consulta.texto_pregunta,
                'chunks_recuperados': chunks,
                'contexto_para_claude': contexto_para_claude,
                'fecha_consulta': consulta.fecha.isoformat()
            }

        except Exception as e:
            return {
                'error': 'Error al procesar consulta RAG',
                'detalles': str(e)
            }

    async def obtener_contexto_objeto(
        self,
        id_objeto: Optional[int] = None,
        nombre: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recupera contexto científico completo de un objeto astronómico.
        
        Pipeline:
        1. Valida que se proporcione id_objeto o nombre (al menos uno)
        2. Si se proporciona nombre, busca el objeto por nombre
        3. Recupera objeto astronómico base
        4. Obtiene documentos asociados
        5. Si es planeta: obtiene características ambientales y evaluaciones
        6. Retorna dict estructurado con toda la información
        
        Args:
            id_objeto: ID del objeto astronómico (usar si se conoce el ID)
            nombre: Nombre del objeto (se busca por nombre si no se proporciona ID)
        
        Returns:
            dict con estructura:
            {
                'objeto': {
                    'id_objeto': int,
                    'nombre': str,
                    'descripcion_cientifica': str
                },
                'documentos': [
                    {
                        'id_doc': int,
                        'titulo': str,
                        'fuente': str,
                        'idioma': str,
                        'fecha': str (ISO) | None
                    },
                    ...
                ],
                'caracteristicas_ambientales': [  # Lista vacía si no es planeta
                    {
                        'tipo': str,
                        'valor': str
                    },
                    ...
                ]
            }
            
            En caso de error:
            {
                'error': str,    # Mensaje descriptivo en español
                'detalles': str  # Información técnica
            }
        
        Raises:
            (No lanza excepciones, retorna dict con 'error' en su lugar)
        
        Example:
            >>> tools = ToolsConsultaRAG(codificador=CodificadorTexto())
            
            # Por ID
            >>> resultado = await tools.obtener_contexto_objeto(id_objeto=1)
            
            # Por nombre
            >>> resultado = await tools.obtener_contexto_objeto(nombre="Tierra")
            >>> resultado['objeto']['id_objeto']
            1
        """
        try:
            # Validar que se proporcione al menos uno
            if not id_objeto and not nombre:
                return {
                    'error': 'Se debe proporcionar id_objeto o nombre',
                    'detalles': 'Al menos uno de los parámetros es obligatorio'
                }

            # Obtener objeto
            objeto = None
            if id_objeto:
                objeto = await self._repo_objetos.obtener_objeto_por_id(id_objeto)
            elif nombre:
                objeto = await self._repo_objetos.obtener_objeto_por_nombre(nombre)

            if not objeto:
                return {
                    'error': 'Objeto astronómico no encontrado',
                    'detalles': f'id={id_objeto}, nombre={nombre}'
                }

            # Obtener documentos asociados al objeto
            documentos = await self._repo_documentos.listar_documentos_por_objeto(
                objeto.id_objeto
            )

            docs_simplificados = [
                {
                    'id_doc': d.id_doc,
                    'titulo': d.titulo,
                    'fuente': d.fuente,
                    'idioma': d.idioma,
                    'fecha': d.fecha.isoformat() if d.fecha else None
                }
                for d in documentos
            ]

            # Construir respuesta base
            respuesta: Dict[str, Any] = {
                'objeto': {
                    'id_objeto': objeto.id_objeto,
                    'nombre': objeto.nombre,
                    'descripcion_cientifica': objeto.descripcion_cientifica
                },
                'documentos': docs_simplificados,
                'caracteristicas_ambientales': []
            }

            # Si es planeta, intentar agregar características ambientales
            try:
                caracteristicas = await self._repo_objetos.obtener_caracteristicas_ambientales(
                    objeto.id_objeto
                )
                respuesta['caracteristicas_ambientales'] = [
                    {
                        'tipo': c.tipo,
                        'valor': c.valor
                    }
                    for c in caracteristicas
                ]
            except Exception:
                # El objeto no es un planeta o no tiene características registradas
                pass

            return respuesta

        except Exception as e:
            return {
                'error': 'Error al obtener contexto del objeto',
                'detalles': str(e)
            }

    def obtener_definiciones_tools(self) -> List[Tool]:
        """
        Retorna las definiciones de las tools en formato MCP.
        
        Se utiliza para registrar las tools en el servidor MCP durante
        inicialización. Cada tool incluye nombre, descripción e inputs.
        
        Returns:
            Lista de Tool (formato MCP) con definiciones de rag_query
            y obtener_contexto_objeto
        """
        return [
            Tool(
                name="rag_query",
                description=(
                    "Realiza una consulta RAG recuperando documentos similares de la BD semántica. "
                    "Vectoriza la pregunta y busca chunks de texto con mayor similitud coseno."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texto_pregunta": {
                            "type": "string",
                            "description": "Pregunta del usuario en lenguaje natural"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Número máximo de chunks a recuperar (default: 5)",
                            "default": 5
                        },
                        "estrategia_chunking": {
                            "type": "string",
                            "enum": ["fixed", "sentence", "semantic"],
                            "description": "Estrategia de chunking para filtrar resultados (opcional)"
                        }
                    },
                    "required": ["texto_pregunta"]
                }
            ),
            Tool(
                name="obtener_contexto_objeto",
                description=(
                    "Recupera contexto científico completo de un objeto astronómico. "
                    "Retorna el objeto, documentos asociados, características y evaluaciones."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_objeto": {
                            "type": "integer",
                            "description": "ID del objeto astronómico"
                        },
                        "nombre": {
                            "type": "string",
                            "description": "Nombre del objeto a buscar"
                        }
                    }
                }
            )
        ]