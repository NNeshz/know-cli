# know

CLI en Python que indexa una carpeta de documentos y responde preguntas en
**lenguaje natural** sobre ellos, citando las fuentes. Es un RAG
(Retrieval-Augmented Generation) de terminal.

Replica la funcionalidad estrella del caso de Chattanooga Labeling Systems de
Harmony: hacer que décadas de documentación operativa (manuales, procedimientos,
especificaciones) sean consultables al instante, en vez de buscar a mano o
preguntarle al empleado con más experiencia.

> **Stack:** Python 3.11+, `typer` (CLI), el SDK oficial `google-genai` (Gemini,
> para generación y embeddings), `numpy` (similitud de coseno) y `python-dotenv`
> (leer la API key). Entorno y dependencias con `uv`.

## Cómo funciona (el pipeline RAG)

```
documentos → chunks → índice → recuperación → generación → respuesta (+ fuentes)
```

1. **Indexar:** lee los documentos, los parte en *chunks* (fragmentos con un
   pequeño solape para no cortar ideas) y guarda un índice local.
2. **Recuperar:** ante una pregunta, busca los chunks más relevantes.
3. **Generar:** arma un prompt con esos chunks como contexto y se lo manda a
   Gemini, que responde **solo** con base en ellos y cita de dónde salió. Si la
   respuesta no está en el contexto, dice *"No encontré esa información"* en vez
   de inventar.

La recuperación está detrás de un `Protocol` (interfaz) con dos modos:

- **`lexical`** (por defecto): coincidencia por palabras (TF-IDF), con solo la
  librería estándar. Sin APIs ni costo: ideal para tener el pipeline andando ya.
- **`semantico`**: usa *embeddings* de Gemini para buscar por **significado**, no
  solo por palabras (encuentra "banda" cuando preguntás por "cinta"). Recupera
  mejor las preguntas con sinónimos.

## Requisitos

- Python 3.11 o superior.
- [uv](https://docs.astral.sh/uv/) para el entorno y las dependencias.
- Una API key **gratuita** de Google Gemini en `GEMINI_API_KEY`
  (obtenela en <https://aistudio.google.com/apikey>, sin tarjeta).

La key hace falta para **generar** respuestas (`preguntar`) y para el **modo
semántico** (que calcula embeddings). El modo léxico de `buscar`/`indexar`
funciona sin ninguna key.

```bash
# En tu terminal, o en un archivo .env en la raíz (ya está en .gitignore):
GEMINI_API_KEY=AIza...

# Opcionales (tienen default en el código):
KNOW_MODEL=gemini-2.5-flash             # modelo de generación
KNOW_EMBEDDINGS_MODEL=gemini-embedding-001  # modelo de embeddings
```

> **Nunca** pongas la key en el código ni la subas al repo. Usá una variable de
> entorno o el archivo `.env` local.

## Instalación y uso

```bash
# Crea el entorno e instala dependencias desde pyproject.toml
uv sync

# 1) Indexar una carpeta de documentos (modo léxico, sin costo)
uv run know indexar ./tests/data/docs

# 2) Preguntar en lenguaje natural (genera con Gemini; cita las fuentes)
uv run know preguntar "¿cada cuánto se cambia el rollo de etiqueta?"

# 3) (Depuración) ver qué fragmentos recupera, sin generar respuesta
uv run know buscar "etiqueta" --k 5

# Modo semántico (calcula embeddings; requiere GEMINI_API_KEY)
uv run know indexar ./tests/data/docs --modo semantico
uv run know preguntar "¿con qué frecuencia se reemplaza la bobina?" --modo semantico
```

## Subcomandos y flags

| Subcomando  | Qué hace                                                   |
| ----------- | ---------------------------------------------------------- |
| `indexar`   | Construye y guarda el índice a partir de una carpeta.      |
| `preguntar` | Recupera contexto y genera una respuesta citada con Gemini.|
| `buscar`    | Solo recuperación (muestra los chunks); útil para depurar. |

| Flag        | Por defecto | Qué hace                                      |
| ----------- | ----------- | --------------------------------------------- |
| `--modo`    | `lexical`   | `lexical` o `semantico`.                      |
| `--k`       | `5`         | Cuántos fragmentos recuperar como contexto.   |
| `--indice`  | `.know`    | Dónde se guarda/lee el índice.                |

## Tests y evaluación

```bash
uv run pytest            # tests unitarios (chunking, coseno, rag mockeado)
uv run python evals.py   # mide el "hit rate" de la recuperación con golden examples
```

Los tests no llaman a ninguna API (la generación se mockea), así que corren
rápido y gratis. `evals.py` usa el modo léxico para correr offline.

## Estructura

```
know/
├── pyproject.toml          # metadatos, dependencias y el comando "know"
├── evals.py                # eval de recuperación (hit rate)
├── src/know/
│   ├── __main__.py         # punto de entrada (python -m know)
│   ├── cli.py              # comandos (typer) + flag --modo
│   ├── docs.py             # cargar documentos y partir en chunks
│   ├── index.py            # guardar/cargar el índice (chunks y vectores)
│   ├── retrieve.py         # Protocol Retriever + LexicalRetriever + SemanticRetriever
│   ├── embeddings.py       # embeddings con Gemini (modo semántico)
│   ├── llm.py              # generación con Gemini (SDK google-genai)
│   └── rag.py              # orquesta: recuperar → armar contexto → generar
└── tests/
    ├── data/docs/          # documentos de ejemplo para probar
    └── test_*.py           # pruebas con pytest
```

La pieza central es el `Protocol` `Retriever`:

```python
from typing import Protocol
from know.docs import Chunk

class Retriever(Protocol):
    def indexar(self, chunks: list[Chunk]) -> None: ...
    def buscar(self, consulta: str, k: int) -> list[Chunk]: ...
```

Cambiar de búsqueda por palabras a búsqueda semántica es cambiar la
**implementación**, no el resto del programa: `rag.py` no se entera. Ese es el punto.

## Alcance honesto

Es un proyecto de aprendizaje, no un sistema de producción. Lo que **sí** hace:
indexar `.txt` y `.md`, recuperar por palabra (TF-IDF) o por embeddings, generar
respuestas citadas sin alucinar, y medir la recuperación con un eval. Lo que
**no** hace todavía (próximos pasos): PDFs, índice en base vectorial, re-ranking.

## Ideas para crecerlo

- **Persistir en Postgres + `pgvector`:** reemplazar el índice en archivo por una
  tabla con columna `vector`. Conecta con el stack de Harmony.
- **Soporte de PDF** para manuales reales (`pypdf`).
- **Búsqueda híbrida:** combinar léxico (exactitud en códigos/nombres) y semántico
  (sinónimos) para lo mejor de ambos.
- **Citas precisas:** que la respuesta señale documento y fragmento exacto.
- **Caché y control de latencia** en las llamadas a la API.
