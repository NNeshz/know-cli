"""Embeddings con Gemini para el modo semántico.

Un *embedding* convierte un texto en un vector (lista de números) que captura su
SIGNIFICADO. Textos parecidos en significado dan vectores cercanos. Usamos un
`task_type` distinto al indexar (documentos) que al consultar (preguntas), porque
el modelo está entrenado para acercar una pregunta a su respuesta.
"""

import os

from google import genai
from google.genai import types

# Modelo de embeddings (gratuito en el free tier de Gemini). Se puede
# sobreescribir con la variable de entorno SABER_EMBEDDINGS_MODEL.
MODELO_EMBEDDINGS = os.environ.get("SABER_EMBEDDINGS_MODEL", "gemini-embedding-001")

# Mapea nuestro 'tipo' simple al task_type que espera la API.
_TASK_TYPE = {
    "document": "RETRIEVAL_DOCUMENT",  # al indexar los documentos
    "query": "RETRIEVAL_QUERY",  # al embeber la pregunta del usuario
}


def embeber(textos: list[str], tipo: str) -> list[list[float]]:
    """Convierte una lista de textos en sus vectores.

    `tipo` es 'document' (al indexar) o 'query' (al consultar). Devuelve un vector
    por cada texto, en el mismo orden.
    """
    if tipo not in _TASK_TYPE:
        raise ValueError(f"tipo debe ser 'document' o 'query'; recibí {tipo!r}.")
    if not textos:
        return []

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta la variable de entorno GEMINI_API_KEY. "
            "Ponela en un archivo .env o expórtala en tu terminal."
        )

    cliente = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60_000),
    )

    try:
        resultado = cliente.models.embed_content(
            model=MODELO_EMBEDDINGS,
            contents=textos,  # type: ignore[arg-type]
            config=types.EmbedContentConfig(task_type=_TASK_TYPE[tipo]),
        )
    except genai.errors.APIError as err:
        raise RuntimeError(f"Error al generar embeddings con Gemini: {err}") from err

    if not resultado.embeddings:
        raise RuntimeError("La API no devolvió embeddings.")

    vectores: list[list[float]] = []
    for emb in resultado.embeddings:
        if emb.values is None:
            raise RuntimeError("Un embedding vino vacío.")
        vectores.append(list(emb.values))
    return vectores
