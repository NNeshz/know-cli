"""Orquestación del pipeline RAG: recuperar contexto y luego generar.

`rag` es el "director de orquesta": no sabe CÓMO se recupera (eso lo decide el
Retriever) ni CÓMO se genera (eso lo decide el Generador). Solo coordina los
pasos. Por eso recibe ambos como parámetros (inyección de dependencias): mañana
puedes pasarle otro buscador u otro proveedor sin tocar esta función.
"""

from know.docs import Chunk
from know.llm import ClaudeGenerador, Generador
from know.retrieve import Retriever


def responder_pregunta(
    pregunta: str,
    retriever: Retriever,
    k: int = 5,
    modelo: str | None = None,
    generador: Generador | None = None,
) -> tuple[str, list[Chunk]]:
    """Recupera el contexto relevante y genera la respuesta.

    Devuelve la respuesta y los chunks usados, para poder citar las fuentes. Si no
    se pasa `generador`, usa Claude.
    """
    generador = generador or ClaudeGenerador()
    contexto = retriever.buscar(pregunta, k)
    respuesta = generador.responder(pregunta, contexto, modelo)
    return respuesta, contexto
