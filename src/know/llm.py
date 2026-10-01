"""Generación de la respuesta con Claude (SDK oficial de Anthropic).

Este módulo es el último eslabón del pipeline:
    ... → recuperar → [generar] → respuesta

Define el contrato `Generador` (un Protocol, igual que `Retriever`) y su
implementación con Claude. Así `rag.py` no depende del proveedor del modelo.

El contexto recuperado y las reglas viven en el *system prompt*; en el mensaje del
usuario va solo la pregunta. La instrucción anti-invención está acá: es lo que
evita que el modelo alucine.
"""

import os
from typing import Protocol

import anthropic

from know.docs import Chunk

# Modelo por defecto: Claude Sonnet actual. Se puede sobrescribir con la variable
# de entorno KNOW_MODEL.
MODELO_POR_DEFECTO = "claude-sonnet-5-5"

TIMEOUT_SEGUNDOS = 60.0
MAX_TOKENS = 1000

# Las reglas contra las alucinaciones. Se le dice al modelo, sin ambigüedad, que
# responda SOLO con el contexto, que cite el archivo y que admita cuando no sabe.
REGLAS = (
    "Eres un asistente que responde preguntas ÚNICAMENTE con base en el CONTEXTO "
    "que se te entrega. Reglas estrictas:\n"
    "- Usa solo información presente de forma explícita en el contexto.\n"
    "- Cita la fuente (el nombre del archivo entre corchetes, p. ej. [manual.md]) "
    "de donde sale cada dato.\n"
    "- Si la respuesta no está en el contexto, responde EXACTAMENTE: "
    "'No encontré esa información.'\n"
    "- No inventes datos ni uses conocimiento externo.\n"
    "- Responde en español, de forma breve y directa."
)


class Generador(Protocol):
    """Contrato que todo generador de respuestas debe cumplir (tipado estructural).

    Igual que `Retriever`: cualquier clase con este método ES un Generador, sin
    heredar de nada. Cambiar de proveedor es cambiar la implementación.
    """

    def responder(
        self, pregunta: str, contexto: list[Chunk], modelo: str | None = None
    ) -> str: ...


def _formatear_contexto(contexto: list[Chunk]) -> str:
    """Une los chunks en un solo bloque de texto, etiquetando cada fuente."""
    return "\n\n".join(f"[{c.fuente}] {c.texto}" for c in contexto)


def construir_system(contexto: list[Chunk]) -> str:
    """Arma el system prompt: las reglas y, debajo, el contexto recuperado."""
    return f"{REGLAS}\n\nCONTEXTO:\n{_formatear_contexto(contexto)}"


class ClaudeGenerador:
    """Genera respuestas con la Messages API de Anthropic.

    El cliente se puede inyectar (útil para pruebas, sin llamadas reales). Si no se
    pasa, se crea uno con la API key de ANTHROPIC_API_KEY. El SDK ya reintenta con
    backoff los errores transitorios (429, 5xx), así que no se repite acá.
    """

    def __init__(self, cliente: anthropic.Anthropic | None = None) -> None:
        self._cliente = cliente

    def _obtener_cliente(self) -> anthropic.Anthropic:
        if self._cliente is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Falta la variable de entorno ANTHROPIC_API_KEY. "
                    "Ponla en un archivo .env o expórtala en tu terminal."
                )
            self._cliente = anthropic.Anthropic(
                api_key=api_key, timeout=TIMEOUT_SEGUNDOS
            )
        return self._cliente

    def responder(
        self, pregunta: str, contexto: list[Chunk], modelo: str | None = None
    ) -> str:
        """Genera una respuesta a `pregunta` usando solo `contexto`."""
        cliente = self._obtener_cliente()
        modelo = modelo or os.environ.get("KNOW_MODEL", MODELO_POR_DEFECTO)

        try:
            respuesta = cliente.messages.create(
                model=modelo,
                max_tokens=MAX_TOKENS,
                system=construir_system(contexto),
                messages=[{"role": "user", "content": pregunta}],
                # Q&A sobre fragmentos: poco razonamiento basta y sale más barato.
                output_config={"effort": "low"},
            )
        except anthropic.AuthenticationError as err:
            raise RuntimeError(
                "ANTHROPIC_API_KEY inválida o sin permisos. Revisa tu key en "
                "https://console.anthropic.com/."
            ) from err
        except anthropic.RateLimitError as err:
            raise RuntimeError(
                "Límite de peticiones de la API de Anthropic alcanzado (rate limit). "
                "Espera un momento y vuelve a intentar."
            ) from err
        except anthropic.APIConnectionError as err:
            raise RuntimeError(
                "No se pudo conectar con la API de Anthropic. "
                "Revisa tu conexión a internet."
            ) from err
        except anthropic.APIStatusError as err:
            raise RuntimeError(
                f"Error de la API de Anthropic ({err.status_code}): {err.message}"
            ) from err

        if respuesta.stop_reason == "refusal":
            raise RuntimeError("El modelo rechazó responder a esta solicitud.")

        # La respuesta es una lista de bloques; nos quedamos solo con los de texto.
        texto = "".join(b.text for b in respuesta.content if b.type == "text")
        if not texto:
            raise RuntimeError("El modelo no devolvió texto en la respuesta.")
        return texto
