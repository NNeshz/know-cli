"""Generación de la respuesta con el modelo de Google Gemini.

Este módulo es el último eslabón del pipeline:
    ... → recuperar → [generar] → respuesta

Recibe la pregunta y el contexto (los chunks recuperados) y le pide al modelo
una respuesta basada SOLO en ese contexto. La instrucción anti-invención vive
acá, en el prompt: es lo que evita que el modelo alucine.
"""

import os
import time

from google import genai
from google.genai import types

from saber.docs import Chunk

# Modelo por defecto (gratuito en el free tier de Gemini). Se puede sobreescribir
# con la variable de entorno SABER_MODEL.
MODELO_POR_DEFECTO = "gemini-2.5-flash"

# Reintentos ante errores transitorios de la API (picos de demanda, rate limits).
REINTENTOS = 3
ESPERA_INICIAL = 2.0  # segundos; el backoff DUPLICA la espera en cada intento

# La regla de oro contra las alucinaciones. Le decimos al modelo, sin ambigüedad,
# que responda SOLO con el contexto y que admita cuando no sabe.
INSTRUCCION_SISTEMA = (
    "Sos un asistente que responde preguntas ÚNICAMENTE con base en el CONTEXTO "
    "que se te entrega. Reglas estrictas:\n"
    "- Usá solo información presente de forma explícita en el contexto.\n"
    "- Si la respuesta no está en el contexto, respondé EXACTAMENTE: "
    "'No encontré esa información.'\n"
    "- No inventes datos ni uses conocimiento externo.\n"
    "- Respondé en español, de forma breve y directa."
)


def _formatear_contexto(contexto: list[Chunk]) -> str:
    """Une los chunks en un solo bloque de texto, etiquetando cada fuente."""
    return "\n\n".join(f"[{c.fuente}] {c.texto}" for c in contexto)


def _es_transitorio(err: genai.errors.APIError) -> bool:
    """True si conviene reintentar: rate limit (429) o error de servidor (5xx)."""
    codigo = getattr(err, "code", None)
    return codigo == 429 or (isinstance(codigo, int) and 500 <= codigo < 600)


def _generar_con_reintentos(
    cliente: genai.Client, modelo: str, prompt: str
) -> types.GenerateContentResponse:
    """Llama a la API reintentando con backoff exponencial los errores transitorios.

    Un error transitorio (un pico de demanda como el 503 que vimos) se reintenta
    esperando cada vez más (2s, 4s, 8s...). Un error definitivo (key inválida) se
    lanza enseguida: no tiene sentido reintentarlo.
    """
    for intento in range(1, REINTENTOS + 1):
        try:
            return cliente.models.generate_content(
                model=modelo,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=INSTRUCCION_SISTEMA,
                    max_output_tokens=1000,
                    temperature=0.0,  # 0 = determinista, pegado al contexto
                ),
            )
        except genai.errors.APIError as err:
            es_ultimo = intento == REINTENTOS
            if es_ultimo or not _es_transitorio(err):
                raise RuntimeError(
                    f"Error al llamar a la API de Gemini: {err}"
                ) from err
            time.sleep(ESPERA_INICIAL * 2 ** (intento - 1))

    # Inalcanzable (el último intento siempre retorna o lanza); tranquiliza a mypy.
    raise RuntimeError("No se obtuvo respuesta del modelo tras los reintentos.")


def responder(pregunta: str, contexto: list[Chunk], modelo: str | None = None) -> str:
    """Genera una respuesta a `pregunta` usando solo `contexto`.

    Lee la API key de la variable de entorno GEMINI_API_KEY (nunca del código).
    Falla temprano y claro si la key no está configurada.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta la variable de entorno GEMINI_API_KEY. "
            "Ponela en un archivo .env o expórtala en tu terminal."
        )

    modelo = modelo or os.environ.get("SABER_MODEL", MODELO_POR_DEFECTO)

    # Cliente con timeout explícito (en milisegundos), como pide el CLAUDE.md.
    cliente = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60_000),
    )

    prompt = f"CONTEXTO:\n{_formatear_contexto(contexto)}\n\nPREGUNTA: {pregunta}"

    respuesta = _generar_con_reintentos(cliente, modelo, prompt)

    texto = respuesta.text
    if texto is None:
        raise RuntimeError("El modelo no devolvió texto en la respuesta.")
    return texto
