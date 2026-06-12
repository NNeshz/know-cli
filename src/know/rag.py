"""Orquestación del pipeline RAG: recuperar contexto y luego generar.

`rag` es el "director de orquesta": no sabe CÓMO se recupera (eso lo decide el
Retriever) ni CÓMO se genera (eso lo decide llm.responder). Solo coordina los
pasos. Por eso recibe el `Retriever` como parámetro (inyección de dependencias):
mañana podés pasarle el léxico o el semántico sin tocar esta función.
"""

from know.docs import Chunk
from know.llm import responder
from know.retrieve import Retriever


def responder_pregunta(
    pregunta: str,
    retriever: Retriever,
    k: int = 5,
    modelo: str | None = None,
) -> tuple[str, list[Chunk]]:
    """Recupera el contexto relevante y genera la respuesta.

    Devuelve la respuesta y los chunks usados, para poder citar las fuentes.
    """
    contexto = retriever.buscar(pregunta, k)
    respuesta = responder(pregunta, contexto, modelo)
    return respuesta, contexto
