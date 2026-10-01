"""Servidor MCP de know: expone la búsqueda y las respuestas citadas a Claude.

MCP (Model Context Protocol) es el estándar para que un cliente como Claude Desktop
o Claude Code use herramientas externas. Este servidor habla por stdio: el cliente lo
lanza como proceso hijo y se comunican por stdin/stdout. Por eso NUNCA se debe
imprimir nada en stdout acá (rompería el protocolo); los logs van a stderr.

Las herramientas son funciones normales (se pueden llamar directo en los tests) y
solo delegan en `servicio.py`: no se duplica ninguna lógica del CLI.

El índice se toma de la variable de entorno KNOW_INDEX (por defecto `.know`).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from know import servicio

INDICE_POR_DEFECTO = ".know"

mcp = MCPServer(
    "know",
    instructions=(
        "Consulta documentos internos (manuales, procedimientos) indexados con know. "
        "Usa `preguntar` para obtener una respuesta citada y `buscar` para ver los "
        "fragmentos originales."
    ),
)


def _ruta_indice() -> Path:
    """Ruta del índice: KNOW_INDEX, o `.know` si no está definida."""
    return Path(os.environ.get("KNOW_INDEX", INDICE_POR_DEFECTO))


@mcp.tool()
def buscar(consulta: str, k: int = 5, modo: str = "lexical") -> str:
    """Busca los fragmentos más relevantes para la consulta y devuelve su fuente.

    No genera respuesta: muestra los pasajes originales de los documentos.
    `modo` es 'lexical' (por palabras) o 'semantico' (por significado).
    """
    resultados = servicio.buscar_fragmentos(consulta, k, _ruta_indice(), modo)
    if not resultados:
        return "No encontré fragmentos relevantes para esa consulta."
    return "\n\n".join(
        f"--- {c.fuente} [chunk {c.indice}] ---\n{c.texto}" for c in resultados
    )


@mcp.tool()
def preguntar(pregunta: str, k: int = 5, modo: str = "lexical") -> str:
    """Responde la pregunta con base solo en los documentos, citando las fuentes.

    Si la información no está en los documentos, responde "No encontré esa
    información." `modo` es 'lexical' o 'semantico'.
    """
    respuesta, contexto = servicio.responder_con_fuentes(
        pregunta, k, _ruta_indice(), modo
    )
    if contexto:
        respuesta += "\n\nFuentes: " + ", ".join(servicio.nombres_de_fuentes(contexto))
    return respuesta


@mcp.tool()
def indexar(carpeta: str, modo: str = "lexical") -> str:
    """Indexa los .txt y .md de una carpeta (reemplaza el índice actual)."""
    indice = _ruta_indice()
    total = servicio.indexar_carpeta(Path(carpeta), indice, modo)
    return f"Indexé {total} chunks desde '{carpeta}' en '{indice}/'."


def main() -> None:
    """Levanta el servidor por stdio (lo invoca `know mcp` o `know-mcp`)."""
    load_dotenv()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
