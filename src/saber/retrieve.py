"""Recuperación: dada una consulta, encontrar los chunks más relevantes.

Define el *contrato* `Retriever` (con `typing.Protocol`) y una implementación
léxica `LexicalRetriever` basada en TF-IDF, usando solo la librería estándar.

La gracia del Protocol: en la Fase 4 agregaremos un `SemanticRetriever` que
cumpla el MISMO contrato, y ni `cli.py` ni `rag.py` tendrán que cambiar.
"""

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from saber.docs import Chunk
from saber.embeddings import embeber

# Una "palabra" es una secuencia de caracteres alfanuméricos. En Python 3 `\w`
# ya incluye letras acentuadas y la ñ, así que "etiqueta" y "mecánica" tokenizan bien.
_PALABRA = re.compile(r"\w+")


def tokenizar(texto: str) -> list[str]:
    """Parte un texto en palabras en minúscula. Función pura y testeable."""
    return _PALABRA.findall(texto.lower())


class Retriever(Protocol):
    """Contrato que todo buscador debe cumplir (tipado estructural).

    Cualquier clase con estos dos métodos —con estas firmas— ES un Retriever a
    ojos de Python, sin necesidad de heredar de nada.
    """

    def indexar(self, chunks: list[Chunk]) -> None: ...
    def buscar(self, consulta: str, k: int) -> list[Chunk]: ...


class LexicalRetriever:
    """Buscador léxico con un esquema TF-IDF simple (solo librería estándar)."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []  # frecuencia de términos por chunk
        self._idf: dict[str, float] = {}  # peso de cada término en el corpus

    def indexar(self, chunks: list[Chunk]) -> None:
        """Cuenta términos por chunk (TF) y el peso de cada término (IDF)."""
        self._chunks = chunks
        self._tf = [Counter(tokenizar(c.texto)) for c in chunks]

        n = len(chunks)
        # Document frequency: en cuántos chunks aparece cada término (una vez por chunk).
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())

        # IDF suavizado (estilo scikit-learn): términos raros pesan más; los que
        # están en todos los chunks pesan poco. El +1 evita divisiones por cero.
        self._idf = {
            termino: math.log((1 + n) / (1 + d)) + 1 for termino, d in df.items()
        }

    def buscar(self, consulta: str, k: int) -> list[Chunk]:
        """Puntúa cada chunk contra la consulta y devuelve los `k` mejores."""
        terminos = tokenizar(consulta)

        puntuados: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            puntaje = sum(tf[t] * self._idf.get(t, 0.0) for t in terminos)
            puntuados.append((puntaje, i))

        # Ordena por puntaje descendente; ante empate, por índice ascendente.
        puntuados.sort(key=lambda par: (-par[0], par[1]))

        # Solo devolvemos chunks con algún solapamiento (puntaje > 0).
        return [self._chunks[i] for puntaje, i in puntuados[:k] if puntaje > 0]


def coseno(a: list[float], b: list[float]) -> float:
    """Similitud de coseno entre dos vectores: 1 = misma dirección, 0 = sin relación.

    Mide el ÁNGULO entre los vectores (su dirección), no su tamaño. Por eso un
    texto largo y uno corto sobre el mismo tema dan similitud alta.
    """
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    normas = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if normas == 0.0:
        return 0.0
    return float(va @ vb) / normas


class SemanticRetriever:
    """Buscador semántico: compara por significado usando embeddings y coseno.

    Cumple el MISMO contrato `Retriever` que `LexicalRetriever`, aunque por dentro
    guarda datos distintos (vectores en vez de conteos de palabras). Por eso
    `cli.py` y `rag.py` lo usan sin cambiar nada: el contrato define el QUÉ, no el
    CÓMO interno.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectores: list[list[float]] = []

    @classmethod
    def desde_vectores(
        cls, chunks: list[Chunk], vectores: list[list[float]]
    ) -> "SemanticRetriever":
        """Crea el retriever con vectores YA calculados (cargados del índice).

        Evita re-embeber los documentos en cada búsqueda: al buscar solo se embebe
        la consulta (1 llamada), no todo el corpus.
        """
        retriever = cls()
        retriever._chunks = chunks
        retriever._vectores = vectores
        return retriever

    def indexar(self, chunks: list[Chunk]) -> None:
        """Embebe los chunks (tipo 'document') y guarda sus vectores."""
        self._chunks = chunks
        self._vectores = embeber([c.texto for c in chunks], "document")

    def buscar(self, consulta: str, k: int) -> list[Chunk]:
        """Embebe la consulta (tipo 'query') y devuelve los k chunks más cercanos."""
        if not self._vectores:
            return []
        vector_consulta = embeber([consulta], "query")[0]
        puntuados = [
            (coseno(vector_consulta, vec), i) for i, vec in enumerate(self._vectores)
        ]
        puntuados.sort(key=lambda par: (-par[0], par[1]))
        return [self._chunks[i] for _, i in puntuados[:k]]
