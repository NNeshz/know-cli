"""Tests del generador con Claude, sin llamadas reales a la API.

Inyectamos un cliente de Anthropic falso: registra lo que se le manda y devuelve
respuestas fijas (o lanza los errores del SDK) para probar cada rama.
"""

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from know import llm
from know.docs import Chunk
from know.llm import ClaudeGenerador, construir_system

CONTEXTO = [Chunk("El rollo se cambia cada 4 horas.", "procedimiento.md", 0)]


def _respuesta(
    texto: str = "Cada 4 horas [procedimiento.md].", stop_reason: str = "end_turn"
) -> SimpleNamespace:
    bloques = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text=texto),
    ]
    return SimpleNamespace(content=bloques, stop_reason=stop_reason)


class ClienteFalso:
    """Imita `anthropic.Anthropic`: expone `.messages.create(...)`."""

    def __init__(self, respuesta: Any = None, error: Exception | None = None) -> None:
        self.llamadas: list[dict[str, Any]] = []
        self._respuesta = respuesta if respuesta is not None else _respuesta()
        self._error = error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.llamadas.append(kwargs)
        if self._error:
            raise self._error
        return self._respuesta


def _generador(cliente: ClienteFalso) -> ClaudeGenerador:
    return ClaudeGenerador(cliente=cliente)  # type: ignore[arg-type]


def _error_http(clase: type[anthropic.APIStatusError], codigo: int) -> Exception:
    peticion = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    respuesta = httpx2.Response(codigo, request=peticion)
    return clase("error de prueba", response=respuesta, body=None)


def test_responder_pone_contexto_y_reglas_en_el_system() -> None:
    cliente = ClienteFalso()

    texto = _generador(cliente).responder("¿cada cuánto?", CONTEXTO)

    assert texto == "Cada 4 horas [procedimiento.md]."  # ignora el bloque thinking
    llamada = cliente.llamadas[0]
    assert "[procedimiento.md] El rollo se cambia cada 4 horas." in llamada["system"]
    assert "No encontré esa información." in llamada["system"]
    # En el mensaje del usuario va solo la pregunta.
    assert llamada["messages"] == [{"role": "user", "content": "¿cada cuánto?"}]


def test_modelo_por_defecto_y_know_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KNOW_MODEL", raising=False)
    cliente = ClienteFalso()
    _generador(cliente).responder("p", CONTEXTO)
    assert cliente.llamadas[0]["model"] == llm.MODELO_POR_DEFECTO

    monkeypatch.setenv("KNOW_MODEL", "claude-opus-5-5")
    _generador(cliente).responder("p", CONTEXTO)
    assert cliente.llamadas[1]["model"] == "claude-opus-5-5"

    _generador(cliente).responder("p", CONTEXTO, modelo="otro-modelo")
    assert cliente.llamadas[2]["model"] == "otro-modelo"


def test_falta_la_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeGenerador().responder("p", CONTEXTO)


@pytest.mark.parametrize(
    "clase, codigo, esperado",
    [
        (anthropic.AuthenticationError, 401, "inválida"),
        (anthropic.RateLimitError, 429, "rate limit"),
        (anthropic.InternalServerError, 500, "500"),
    ],
)
def test_errores_de_la_api_se_vuelven_mensajes_claros(
    clase: type[anthropic.APIStatusError], codigo: int, esperado: str
) -> None:
    cliente = ClienteFalso(error=_error_http(clase, codigo))
    with pytest.raises(RuntimeError, match=esperado):
        _generador(cliente).responder("p", CONTEXTO)


def test_error_de_conexion() -> None:
    peticion = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    cliente = ClienteFalso(error=anthropic.APIConnectionError(request=peticion))
    with pytest.raises(RuntimeError, match="conectar"):
        _generador(cliente).responder("p", CONTEXTO)


def test_refusal_y_respuesta_sin_texto() -> None:
    with pytest.raises(RuntimeError, match="rechazó"):
        _generador(ClienteFalso(_respuesta(stop_reason="refusal"))).responder(
            "p", CONTEXTO
        )

    vacia = SimpleNamespace(content=[], stop_reason="end_turn")
    with pytest.raises(RuntimeError, match="no devolvió texto"):
        _generador(ClienteFalso(vacia)).responder("p", CONTEXTO)


def test_construir_system_etiqueta_cada_fuente() -> None:
    chunks = [Chunk("uno", "a.md", 0), Chunk("dos", "b.md", 1)]
    system = construir_system(chunks)
    assert "[a.md] uno" in system and "[b.md] dos" in system
