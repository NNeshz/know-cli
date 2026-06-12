"""Tests de la recuperación: coseno (puro) y recuperación léxica."""

import pytest

from know.docs import Chunk
from know.retrieve import LexicalRetriever, coseno, tokenizar


@pytest.mark.parametrize(
    "a, b, esperado",
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),  # misma dirección
        ([1.0, 0.0], [0.0, 1.0], 0.0),  # perpendiculares
        ([1.0, 0.0], [-1.0, 0.0], -1.0),  # opuestos
        ([1.0, 1.0], [2.0, 2.0], 1.0),  # misma dirección, distinto largo
    ],
)
def test_coseno(a: list[float], b: list[float], esperado: float) -> None:
    assert coseno(a, b) == pytest.approx(esperado)


def test_coseno_vector_cero_no_explota() -> None:
    assert coseno([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_tokenizar_normaliza_a_minusculas_y_separa() -> None:
    assert tokenizar("Hola, MUNDO  cruel!") == ["hola", "mundo", "cruel"]


def test_lexical_recupera_el_chunk_correcto() -> None:
    chunks = [
        Chunk("el rollo de etiqueta se cambia cada 4 horas", "etiqueta.md", 0),
        Chunk("la banda transportadora se lubrica los lunes", "mantenimiento.md", 0),
    ]
    retriever = LexicalRetriever()
    retriever.indexar(chunks)

    resultados = retriever.buscar("cada cuanto se cambia el rollo de etiqueta", k=1)

    assert resultados[0].fuente == "etiqueta.md"
