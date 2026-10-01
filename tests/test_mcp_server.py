"""Tests de las herramientas del servidor MCP, llamándolas como funciones normales.

No levantamos el servidor ni usamos la API real: las herramientas son funciones
Python, así que las invocamos directo. El índice va a un directorio temporal
(KNOW_INDEX) y el generador de Claude se reemplaza por uno falso.
"""

import asyncio
from pathlib import Path

import pytest

from know import mcp_server, rag
from know.docs import Chunk

DOCS = Path(__file__).parent / "data" / "docs"


class GeneradorFalso:
    def responder(
        self, pregunta: str, contexto: list[Chunk], modelo: str | None = None
    ) -> str:
        return f"Respuesta a: {pregunta}"


@pytest.fixture
def indice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    destino = tmp_path / "indice"
    monkeypatch.setenv("KNOW_INDEX", str(destino))
    return destino


@pytest.fixture
def indice_listo(indice: Path) -> Path:
    mcp_server.indexar(str(DOCS))
    return indice


def test_indexar_crea_el_indice_en_know_index(indice: Path) -> None:
    mensaje = mcp_server.indexar(str(DOCS))

    assert "Indexé 2 chunks" in mensaje
    assert (indice / "chunks.json").exists()


def test_indexar_carpeta_inexistente(indice: Path) -> None:
    with pytest.raises(NotADirectoryError):
        mcp_server.indexar("/no/existe")


def test_buscar_devuelve_fragmentos_con_su_fuente(indice_listo: Path) -> None:
    salida = mcp_server.buscar("rollo de etiqueta", k=1)

    assert "procedimiento-etiqueta.md" in salida
    assert "cada 4 horas" in salida


def test_buscar_sin_resultados(indice_listo: Path) -> None:
    assert "No encontré fragmentos" in mcp_server.buscar("zzzz qqqq")


def test_buscar_sin_indice_da_error_claro(indice: Path) -> None:
    with pytest.raises(FileNotFoundError, match="know indexar"):
        mcp_server.buscar("algo")


def test_buscar_modo_desconocido(indice_listo: Path) -> None:
    with pytest.raises(ValueError, match="Modo desconocido"):
        mcp_server.buscar("algo", modo="magico")


def test_buscar_semantico_sin_embeddings(indice_listo: Path) -> None:
    with pytest.raises(RuntimeError, match="no tiene embeddings"):
        mcp_server.buscar("algo", modo="semantico")


def test_preguntar_devuelve_respuesta_y_fuentes(
    indice_listo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rag, "ClaudeGenerador", GeneradorFalso)

    salida = mcp_server.preguntar("¿cada cuánto se cambia el rollo de etiqueta?", k=1)

    assert salida.startswith("Respuesta a: ¿cada cuánto")
    assert "Fuentes: procedimiento-etiqueta.md" in salida


def test_las_tres_herramientas_estan_registradas() -> None:
    herramientas = asyncio.run(mcp_server.mcp.list_tools())
    nombres = {t.name for t in herramientas}
    assert nombres == {"buscar", "preguntar", "indexar"}
