"""Eval simple de la recuperación: mide el *hit rate* con golden examples.

Un *eval* no pregunta "¿el código corre?" (eso lo hacen los tests), sino "¿qué tan
BIEN recupera el sistema?". Corre una lista de preguntas con su documento esperado
y cuenta en cuántas la recuperación trajo el chunk correcto.

Usa el modo léxico a propósito: así corre offline, sin API key ni costo.

Uso:  uv run python evals.py
"""

from pathlib import Path

from saber.docs import cargar_chunks
from saber.retrieve import LexicalRetriever

CARPETA_DOCS = Path("tests/data/docs")

# (pregunta, archivo que DEBERÍA recuperarse primero)
GOLDEN: list[tuple[str, str]] = [
    ("¿cada cuánto se cambia el rollo de etiqueta?", "procedimiento-etiqueta.md"),
    ("¿qué pasa con una falla mecánica larga?", "mantenimiento.md"),
    ("¿cuándo se lubrica la banda transportadora?", "mantenimiento.md"),
]


def main() -> None:
    chunks = cargar_chunks(CARPETA_DOCS)
    retriever = LexicalRetriever()
    retriever.indexar(chunks)

    aciertos = 0
    for pregunta, esperado in GOLDEN:
        resultados = retriever.buscar(pregunta, k=1)
        recuperado = resultados[0].fuente if resultados else "(nada)"
        ok = recuperado == esperado
        aciertos += int(ok)
        marca = "OK  " if ok else "FALLO"
        print(
            f"[{marca}] {pregunta}\n        esperado={esperado} | recuperado={recuperado}"
        )

    print(f"\nHit rate: {aciertos}/{len(GOLDEN)}")


if __name__ == "__main__":
    main()
