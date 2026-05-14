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
from datetime import datetime, date
import re
from mcp.types import Tool, TextContent

from database.repositorio_objetos import RepositorioObjetos
from database.repositorio_documentos import RepositorioDocumentos
from database.repositorio_observaciones import RepositorioObservaciones
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
        self._repo_observaciones = RepositorioObservaciones()

    def obtener_definiciones_tools(self) -> List[Tool]:
        """
        Retorna las definiciones de herramientas MCP para registro en el servidor.
        
        Genera Tool objects que describen a Claude la interfaz de cada herramienta
        de gestión de objetos, incluyendo nombres, descripciones, y esquemas
        de entrada JSON.
        
        Returns:
            List[Tool] con definiciones de herramientas MCP para operaciones CRUD
        
        Estructura de cada Tool:
            - name: Identificador de herramienta para Claude
            - description: Descripción en español de qué hace
            - inputSchema: JSON Schema describiendo los parámetros
        """
        return [
            Tool(
                name="crear_objeto_astronomico",
                description="Crea un nuevo objeto astronómico (galaxia, sistema, estrella, planeta, luna) con embeddings automáticos.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tipo": {
                            "type": "string",
                            "enum": ["galaxia", "sistema_estelar", "estrella", "planeta", "luna"],
                            "description": "Tipo de objeto astronómico"
                        },
                        "nombre": {
                            "type": "string",
                            "description": "Nombre del objeto"
                        },
                        "descripcion_cientifica": {
                            "type": "string",
                            "description": "Descripción científica del objeto"
                        },
                        "atributos": {
                            "type": "object",
                            "description": "Atributos específicos por tipo"
                        }
                    },
                    "required": ["tipo", "nombre", "descripcion_cientifica"]
                }
            ),
            Tool(
                name="obtener_objeto_astronomico",
                description="Obtiene un objeto astronómico por ID o nombre con sus detalles completos.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_objeto": {
                            "type": "integer",
                            "description": "ID del objeto"
                        },
                        "nombre": {
                            "type": "string",
                            "description": "Nombre del objeto"
                        }
                    }
                }
            ),
            Tool(
                name="crear_documento_con_embeddings",
                description="Crea un documento científico con chunking automático y embeddings para cada chunk.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "titulo": {
                            "type": "string",
                            "description": "Título del documento"
                        },
                        "contenido_texto": {
                            "type": "string",
                            "description": "Contenido completo del documento"
                        },
                        "id_objeto": {
                            "type": "integer",
                            "description": "ID del objeto astronómico relacionado"
                        },
                        "estrategia_chunking": {
                            "type": "string",
                            "enum": ["sentence", "paragraph", "fixed"],
                            "description": "Estrategia de división de texto"
                        }
                    },
                    "required": ["titulo", "contenido_texto", "id_objeto"]
                }
            ),
            Tool(
                name="crear_imagen_con_embedding",
                description="Crea un registro de imagen astronómica con embedding CLIP multimodal.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ruta_archivo": {
                            "type": "string",
                            "description": "Ruta al archivo de imagen"
                        },
                        "descripcion": {
                            "type": "string",
                            "description": "Descripción textual de la imagen"
                        },
                        "etiquetas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Etiquetas de categorización"
                        },
                        "id_doc": {
                            "type": "integer",
                            "description": "ID del documento asociado"
                        }
                    },
                    "required": ["ruta_archivo", "descripcion", "id_doc"]
                }
            )
        ]