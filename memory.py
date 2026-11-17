"""Compatibilidad: el código vive en el paquete `agentic_memory_layer`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `memory`
directamente."""

from agentic_memory_layer.memory import *  # noqa: F401,F403
from agentic_memory_layer.memory import AgenticMemory, VectorStore  # noqa: F401
