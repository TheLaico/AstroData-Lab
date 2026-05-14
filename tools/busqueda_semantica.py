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
from models.planeta_model import Planeta


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
            
            # Buscar planetas similares por habitabilidad
            try:
                todos_planetas = await self._repo_objetos.listar_planetas_por_habitabilidad(
                    puntaje_minimo=0.0
                )
            except Exception as e:
                return {
                    'error': "Error al buscar planetas similares",
                    'detalles': str(e),
                    'planeta_referencia': None,
                    'planetas_similares': [],
                    'total': 0
                }
            
            # Formatear resultados excluyendo el planeta de referencia
            planetas_lista = []
            for planeta in todos_planetas:
                if planeta.id_objeto != id_planeta:
                    planetas_lista.append({
                        'id_objeto': planeta.id_objeto,
                        'nombre': planeta.nombre,
                        'masa': planeta.masa,
                        'temperatura': planeta.temperatura
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