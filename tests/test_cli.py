"""Tests del CLI y de los mensajes de error (sin APIs reales)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from know import embeddings
from know.cli import app

DOCS = Path(__file__).parent / "data" / "docs"
runner = CliRunner()


def test_lexico_y_buscar_funcionan_sin_ninguna_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    indice = str(tmp_path / "idx")

    assert runner.invoke(app, ["indexar", str(DOCS), "--indice", indice]).exit_code == 0
    resultado = runner.invoke(app, ["buscar", "etiqueta", "--indice", indice])

    assert resultado.exit_code == 0
    assert "procedimiento-etiqueta.md" in resultado.output


def test_preguntar_sin_anthropic_api_key_da_error_claro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    indice = str(tmp_path / "idx")
    runner.invoke(app, ["indexar", str(DOCS), "--indice", indice])

    resultado = runner.invoke(app, ["preguntar", "¿algo?", "--indice", indice])

    assert resultado.exit_code == 1
    assert "ANTHROPIC_API_KEY" in resultado.output


def test_embeddings_sin_gemini_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        embeddings.embeber(["hola"], "document")


def test_modo_desconocido_sale_con_error(tmp_path: Path) -> None:
    resultado = runner.invoke(
        app, ["indexar", str(DOCS), "--indice", str(tmp_path), "--modo", "magico"]
    )
    assert resultado.exit_code == 1
    assert "Modo desconocido" in resultado.output
