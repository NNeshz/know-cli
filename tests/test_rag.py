"""Test de la orquestación RAG sin pegarle a ninguna API real.

Fingimos el Retriever y el Generador. Como `rag` solo conoce los contratos
`Retriever` y `Generador`, le pasamos dobles de prueba que los cumplan. Así el test
es rápido y gratis.
"""

from know import rag
from know.docs import Chunk


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


class GeneradorFalso:
    """Doble de prueba: cumple el contrato Generador y registra lo que recibe."""

    def __init__(self) -> None:
        self.llamadas: list[tuple[str, list[Chunk], str | None]] = []

    def responder(
        self, pregunta: str, contexto: list[Chunk], modelo: str | None = None
    ) -> str:
        self.llamadas.append((pregunta, contexto, modelo))
        return "respuesta de prueba"


def test_responder_pregunta_recupera_y_genera() -> None:
    contexto_esperado = [Chunk("contenido relevante", "doc.md", 0)]
    retriever = RetrieverFalso(contexto_esperado)
    generador = GeneradorFalso()

    respuesta, contexto = rag.responder_pregunta(
        "una pregunta", retriever, k=3, generador=generador
    )

    assert respuesta == "respuesta de prueba"
    assert contexto == contexto_esperado
    assert retriever.consultas == ["una pregunta"]  # de verdad usó el retriever
    assert generador.llamadas == [("una pregunta", contexto_esperado, None)]
