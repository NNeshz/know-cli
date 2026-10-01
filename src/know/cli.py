"""Definición del CLI con typer: la app y sus subcomandos.

El flag `--modo lexical|semantico` elige la implementación del Retriever. Fíjate
que `rag.py` no cambió nada al agregar el modo semántico: el único lugar que decide
qué Retriever instanciar está en `servicio.py`. Ese es el pago del Protocol.

El CLI es solo una "puerta de entrada": parsea argumentos, llama a `servicio.py` y
presenta el resultado. El servidor MCP (`mcp_server.py`) usa las mismas funciones.
"""

from pathlib import Path

import typer
from dotenv import load_dotenv

from know import servicio

# Carga las variables del archivo .env (p. ej. ANTHROPIC_API_KEY) al entorno, si
# existe. Así no dependemos de hacer `export` en cada terminal.
load_dotenv()

# `app` es la aplicación de línea de comandos. Cada función decorada con
# `@app.command()` se vuelve un subcomando (know indexar / preguntar / buscar / mcp).
app = typer.Typer(help="Búsqueda de conocimiento sobre documentos con IA.")

ERRORES_ESPERADOS = (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError)


@app.command()
def indexar(carpeta: str, indice: str = ".know", modo: str = "lexical") -> None:
    """Lee los documentos de CARPETA, los parte en chunks y guarda el índice."""
    try:
        total = servicio.indexar_carpeta(Path(carpeta), Path(indice), modo)
    except ERRORES_ESPERADOS as err:
        typer.secho(f"Error: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from err

    extra = " (con embeddings)" if modo == "semantico" else ""
    typer.echo(f"Indexé {total} chunks desde '{carpeta}' en '{indice}/'{extra}.")


@app.command()
def preguntar(
    pregunta: str, k: int = 5, indice: str = ".know", modo: str = "lexical"
) -> None:
    """Responde PREGUNTA en lenguaje natural citando las fuentes (usa Claude)."""
    try:
        respuesta, contexto = servicio.responder_con_fuentes(
            pregunta, k, Path(indice), modo
        )
    except ERRORES_ESPERADOS as err:
        # Errores esperables (índice faltante, key ausente, rate limit): mensaje
        # limpio, no un traceback gigante. Salimos con código 1 para señalar el fallo.
        typer.secho(f"Error: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from err

    typer.echo(respuesta)
    if contexto:
        typer.echo("\nFuentes: " + ", ".join(servicio.nombres_de_fuentes(contexto)))


@app.command()
def buscar(
    consulta: str, k: int = 5, indice: str = ".know", modo: str = "lexical"
) -> None:
    """Muestra los fragmentos más relevantes para CONSULTA (sin generar respuesta)."""
    try:
        resultados = servicio.buscar_fragmentos(consulta, k, Path(indice), modo)
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


@app.command()
def mcp() -> None:
    """Levanta el servidor MCP por stdio (índice en KNOW_INDEX, default .know)."""
    # Import perezoso: el CLI normal no necesita cargar el SDK de MCP.
    from know.mcp_server import main

    main()


if __name__ == "__main__":
    app()
