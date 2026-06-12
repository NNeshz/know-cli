"""Definición del CLI con typer: la app y sus tres subcomandos.

El flag `--modo lexical|semantico` elige la implementación del Retriever. Fijate
que `rag.py` no cambió nada al agregar el modo semántico: acá, en el "armador",
es el único lugar que decide qué Retriever instanciar. Ese es el pago del Protocol.
"""

from pathlib import Path

import typer
from dotenv import load_dotenv

from know.docs import Chunk, cargar_chunks
from know.embeddings import embeber
from know.index import cargar_indice, cargar_vectores, guardar_indice
from know.rag import responder_pregunta
from know.retrieve import LexicalRetriever, Retriever, SemanticRetriever

# Carga las variables del archivo .env (p. ej. GEMINI_API_KEY) al entorno, si
# existe. Así no dependemos de hacer `export` en cada terminal.
load_dotenv()

# `app` es la aplicación de línea de comandos. Cada función decorada con
# `@app.command()` se vuelve un subcomando (know indexar / preguntar / buscar).
app = typer.Typer(help="Búsqueda de conocimiento sobre documentos con IA.")

ERRORES_ESPERADOS = (FileNotFoundError, NotADirectoryError, RuntimeError)


def _construir_retriever(modo: str, chunks: list[Chunk], indice: Path) -> Retriever:
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
                "El índice no tiene embeddings. Reindexá con: "
                f"know indexar <carpeta> --indice {indice} --modo semantico"
            )
        return SemanticRetriever.desde_vectores(chunks, vectores)
    raise typer.BadParameter(
        f"Modo desconocido: {modo!r}. Usá 'lexical' o 'semantico'."
    )


@app.command()
def indexar(carpeta: str, indice: str = ".know", modo: str = "lexical") -> None:
    """Lee los documentos de CARPETA, los parte en chunks y guarda el índice."""
    try:
        chunks = cargar_chunks(Path(carpeta))
        # En modo semántico, calculamos los embeddings UNA vez (acá) y los guardamos.
        vectores = (
            embeber([c.texto for c in chunks], "document")
            if modo == "semantico"
            else None
        )
        guardar_indice(chunks, Path(indice), vectores=vectores)
    except ERRORES_ESPERADOS as err:
        typer.secho(f"Error: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from err

    extra = " (con embeddings)" if modo == "semantico" else ""
    typer.echo(f"Indexé {len(chunks)} chunks desde '{carpeta}' en '{indice}/'{extra}.")


@app.command()
def preguntar(
    pregunta: str, k: int = 5, indice: str = ".know", modo: str = "lexical"
) -> None:
    """Responde PREGUNTA en lenguaje natural citando las fuentes (usa IA)."""
    try:
        chunks = cargar_indice(Path(indice))
        retriever = _construir_retriever(modo, chunks, Path(indice))
        respuesta, contexto = responder_pregunta(pregunta, retriever, k)
    except ERRORES_ESPERADOS as err:
        # Errores esperables (índice faltante, fallo de API): mensaje limpio, no
        # un traceback gigante. Salimos con código 1 para señalar el fallo.
        typer.secho(f"Error: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from err

    typer.echo(respuesta)
    if contexto:
        fuentes = sorted({chunk.fuente for chunk in contexto})
        typer.echo("\nFuentes: " + ", ".join(fuentes))


@app.command()
def buscar(
    consulta: str, k: int = 5, indice: str = ".know", modo: str = "lexical"
) -> None:
    """Muestra los fragmentos más relevantes para CONSULTA (sin generar respuesta)."""
    try:
        chunks = cargar_indice(Path(indice))
        retriever = _construir_retriever(modo, chunks, Path(indice))
        resultados = retriever.buscar(consulta, k)
    except ERRORES_ESPERADOS as err:
        typer.secho(f"Error: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from err

    if not resultados:
        typer.echo("No encontré fragmentos relevantes para esa consulta.")
        return

    typer.echo(f"Top {len(resultados)} fragmentos para: {consulta!r}\n")
    for chunk in resultados:
        typer.echo(f"--- {chunk.fuente} [chunk {chunk.indice}] ---")
        typer.echo(chunk.texto)
        typer.echo("")


if __name__ == "__main__":
    app()
