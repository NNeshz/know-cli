"""Tests de la lógica pura de chunking (sin tocar disco)."""

import pytest

from saber.docs import partir_texto


@pytest.mark.parametrize(
    "nchars, tam, esperado",
    [
        (0, 100, 0),  # texto vacío -> ningún chunk
        (100, 100, 1),  # justo del tamaño -> un chunk
        (150, 100, 2),  # uno y medio -> dos chunks
        (300, 100, 3),  # tres tamaños exactos -> tres chunks
    ],
)
def test_partir_cuenta_sin_solape(nchars: int, tam: int, esperado: int) -> None:
    assert len(partir_texto("a" * nchars, tam, solape=0)) == esperado


def test_partir_respeta_tamano_maximo() -> None:
    pedazos = partir_texto("a" * 250, tam=100, solape=10)
    assert all(len(p) <= 100 for p in pedazos)


def test_solape_repite_el_borde() -> None:
    texto = "".join(str(i % 10) for i in range(100))  # 100 caracteres distintos
    pedazos = partir_texto(texto, tam=50, solape=10)
    # El final del primer pedazo debe repetirse al inicio del segundo.
    assert pedazos[0][-10:] == pedazos[1][:10]


@pytest.mark.parametrize(
    "tam, solape",
    [
        (0, 0),  # tam no positivo
        (-5, 0),  # tam negativo
        (100, 100),  # solape == tam (loop infinito si no se valida)
        (100, 150),  # solape > tam
    ],
)
def test_partir_parametros_invalidos(tam: int, solape: int) -> None:
    with pytest.raises(ValueError):
        partir_texto("hola mundo", tam, solape)
