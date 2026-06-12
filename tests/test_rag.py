"""Test de la orquestación RAG sin pegarle a ninguna API real.

Mockeamos (fingimos) el Retriever y la función `responder`. Como `rag` solo
conoce el contrato `Retriever`, le podemos pasar un doble de prueba que lo cumpla,
y reemplazamos `responder` con `monkeypatch`. Así el test es rápido y gratis.
"""

from saber import rag
from saber.docs import Chunk


class RetrieverFalso:
    """Doble de prueba: cumple el contrato Retriever pero devuelve chunks fijos."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.consultas: list[str] = []

    def indexar(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def buscar(self, consulta: str, k: int) -> list[Chunk]:
        self.consultas.append(consulta)
        return self._chunks


def test_responder_pregunta_recupera_y_genera(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    contexto_esperado = [Chunk("contenido relevante", "doc.md", 0)]
    retriever = RetrieverFalso(contexto_esperado)

    # Reemplazamos `responder` (importado dentro de rag) por una versión falsa.
    def responder_falso(
        pregunta: str, contexto: list[Chunk], modelo: str | None = None
    ) -> str:
        return "respuesta de prueba"

    monkeypatch.setattr(rag, "responder", responder_falso)

    respuesta, contexto = rag.responder_pregunta("una pregunta", retriever, k=3)

    assert respuesta == "respuesta de prueba"
    assert contexto == contexto_esperado
    assert retriever.consultas == ["una pregunta"]  # de verdad usó el retriever
