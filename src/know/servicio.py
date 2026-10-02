"""Operaciones de alto nivel que comparten el CLI y el servidor MCP.

Tanto `cli.py` (typer) como `mcp_server.py` (herramientas MCP) son solo "puertas de
entrada": reciben parámetros, llaman a estas funciones y presentan el resultado a su
manera. La lógica vive aquí, una sola vez.
"""

from pathlib import Path

from know.docs import Chunk, cargar_chunks
from know.embeddings import embeber
from know.index import cargar_indice, cargar_vectores, guardar_indice
from know.llm import Generador
from know.rag import responder_pregunta
from know.retrieve import LexicalRetriever, Retriever, SemanticRetriever

MODOS = ("lexical", "semantico")


def construir_retriever(modo: str, chunks: list[Chunk], indice: Path) -> Retriever:
    """Devuelve el Retriever adecuado al `modo`, ya listo para buscar.

    - lexical: cuenta términos en el momento (no necesita el índice de vectores).
    - semantico: reusa los vectores guardados en el índice (no re-embebe el corpus).
    """
    if modo == "lexical":
        retriever: Retriever = LexicalRetriever()
        retriever.indexar(chunks)
        return retriever
    if modo == "semantico":
        vectores = cargar_vectores(indice)
        if vectores is None:
            raise RuntimeError(
                "El índice no tiene embeddings. Reindexa con: "
                f"know indexar <carpeta> --indice {indice} --modo semantico"
            )
        return SemanticRetriever.desde_vectores(chunks, vectores)
    raise ValueError(f"Modo desconocido: {modo!r}. Usa 'lexical' o 'semantico'.")


def indexar_carpeta(carpeta: Path, indice: Path, modo: str = "lexical") -> int:
    """Lee los documentos de `carpeta`, los parte en chunks y guarda el índice.

    Devuelve cuántos chunks se indexaron. En modo semántico calcula los embeddings
    UNA vez (aquí) y los guarda junto con los chunks.
    """
    if modo not in MODOS:
        raise ValueError(f"Modo desconocido: {modo!r}. Usa 'lexical' o 'semantico'.")
    chunks = cargar_chunks(carpeta)
    vectores = (
        embeber([c.texto for c in chunks], "document") if modo == "semantico" else None
    )
    guardar_indice(chunks, indice, vectores=vectores)
    return len(chunks)


def buscar_fragmentos(
    consulta: str, k: int, indice: Path, modo: str = "lexical"
) -> list[Chunk]:
    """Devuelve los `k` fragmentos más relevantes para `consulta` (sin generar)."""
    chunks = cargar_indice(indice)
    return construir_retriever(modo, chunks, indice).buscar(consulta, k)


def responder_con_fuentes(
    pregunta: str,
    k: int,
    indice: Path,
    modo: str = "lexical",
    generador: Generador | None = None,
) -> tuple[str, list[Chunk]]:
    """Recupera contexto y genera la respuesta; devuelve también los chunks usados."""
    chunks = cargar_indice(indice)
    retriever = construir_retriever(modo, chunks, indice)
    return responder_pregunta(pregunta, retriever, k, generador=generador)


def nombres_de_fuentes(contexto: list[Chunk]) -> list[str]:
    """Archivos únicos citados por el contexto, ordenados."""
    return sorted({chunk.fuente for chunk in contexto})
