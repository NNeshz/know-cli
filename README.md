# saber

CLI en Python que indexa una carpeta de documentos y responde preguntas en
**lenguaje natural** sobre ellos, citando las fuentes. Es un RAG
(Retrieval-Augmented Generation) de terminal.

Replica la funcionalidad estrella del caso de Chattanooga Labeling Systems de
Harmony: hacer que décadas de documentación operativa (manuales, procedimientos,
especificaciones) sean consultables al instante, en vez de buscar a mano o
preguntarle al empleado con más experiencia.

> **Por qué Python aquí:** es el ecosistema más maduro para IA. Usamos los SDKs
> oficiales de Anthropic (`anthropic`) y de Voyage (`voyageai`), `numpy` para la
> similitud y `typer` para el CLI. (El proyecto hermano `turno` está en Go;
> usar el lenguaje correcto para cada caso es a propósito.)

## Cómo funciona (el pipeline RAG)

1. **Indexar:** lee los documentos, los parte en *chunks* (fragmentos), y guarda
   una representación de cada uno en un índice local.
2. **Recuperar:** ante una pregunta, busca los chunks más relevantes.
3. **Generar:** arma un prompt con esos chunks como contexto y se lo manda a la
   API de Claude, que responde basándose en ellos y cita de dónde salió.

La recuperación está detrás de un `Protocol` (interfaz) con dos modos:

- **`lexical`** (por defecto): coincidencia por palabras (TF-IDF), sin APIs
  externas ni costo. Sirve para tener el pipeline funcionando ya.
- **`semantico`**: usa *embeddings* para entender el significado, no solo las
  palabras. Recupera mejor pero requiere una API de embeddings.

> Anthropic no ofrece un modelo de embeddings propio; su proveedor recomendado
> es **Voyage AI** (modelo sugerido para retrieval: `voyage-3-large`). El modo
> semántico usa Voyage; el modo léxico no necesita nada.

## Alcance honesto

Esto es un proyecto de aprendizaje, no un sistema de producción. Lo que **sí**
hace: indexar `.txt` y `.md`, recuperar por palabra o por embeddings, y generar
respuestas citadas. Lo que **no** hace todavía (y dejas como próximos pasos):
PDFs, índice persistente en base vectorial, re-ranking, ni evaluación automática
de calidad. Saber qué falta es parte del mérito.

## Requisitos

- Python 3.11 o superior.
- [uv](https://docs.astral.sh/uv/) para manejar el entorno y dependencias
  (o `pip` + `venv` si prefieres).
- Una API key de Anthropic en `ANTHROPIC_API_KEY` (para generar respuestas).
- *Solo para el modo semántico:* una API key de Voyage en `VOYAGE_API_KEY`.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export VOYAGE_API_KEY="pa-..."          # opcional, solo modo semántico
export CLAUDE_MODEL="claude-sonnet-4-6" # opcional; por defecto un modelo actual
```

Nunca pongas las keys en el código ni las subas al repo. Para desarrollo local
puedes usar un archivo `.env` (incluido en `.gitignore`).

## Instalación y uso

```bash
# Con uv: crea el entorno e instala dependencias desde pyproject.toml
uv sync

# 1) Indexar una carpeta de documentos
uv run saber indexar ./tests/data/docs

# 2) Preguntar en lenguaje natural
uv run saber preguntar "¿cuál es el procedimiento para cambiar la etiqueta?"

# 3) (Depuración) ver qué fragmentos recupera, sin generar respuesta
uv run saber buscar "etiqueta" --k 5

# Usar el modo semántico (requiere VOYAGE_API_KEY)
uv run saber indexar ./tests/data/docs --modo semantico
uv run saber preguntar "..." --modo semantico
```

Si usas `pip`: `python -m venv .venv && source .venv/bin/activate && pip install -e .`,
y luego `saber preguntar "..."`.

## Subcomandos y flags

| Subcomando | Qué hace                                              |
| ---------- | ----------------------------------------------------- |
| `indexar`  | Construye el índice a partir de una carpeta.          |
| `preguntar`| Recupera contexto y genera una respuesta con Claude.  |
| `buscar`   | Solo recuperación (muestra los chunks); útil para depurar. |

| Flag        | Por defecto | Qué hace                                      |
| ----------- | ----------- | --------------------------------------------- |
| `--modo`    | `lexical`   | `lexical` o `semantico`.                      |
| `--k`       | `5`         | Cuántos fragmentos recuperar como contexto.   |
| `--indice`  | `.saber/`   | Dónde se guarda/lee el índice.                |

## Estructura

```
saber/
├── pyproject.toml          # metadatos, dependencias y el comando "saber"
├── src/saber/
│   ├── __init__.py
│   ├── __main__.py         # punto de entrada
│   ├── cli.py              # comandos (typer)
│   ├── docs.py             # cargar documentos y partir en chunks
│   ├── index.py            # guardar y cargar el índice en disco
│   ├── retrieve.py         # Protocol Retriever + impls lexical y semantico
│   ├── llm.py              # cliente de la API de Claude (SDK anthropic)
│   └── rag.py              # orquesta: recuperar → armar prompt → generar
├── tests/
│   └── data/docs/          # documentos de ejemplo para probar
└── README.md
```

La pieza central es el `Protocol` `Retriever`:

```python
from typing import Protocol
from saber.docs import Chunk

class Retriever(Protocol):
    def indexar(self, chunks: list[Chunk]) -> None: ...
    def buscar(self, consulta: str, k: int) -> list[Chunk]: ...
```

Cambiar de búsqueda por palabras a búsqueda semántica es cambiar la
implementación, no el resto del programa. Ese es el punto.

## Ideas para crecerlo

- **Persistir en Postgres + `pgvector`:** conecta directo con el primer proyecto
  (`turno`) y con el stack de Harmony. Reemplaza el índice en archivo por una
  tabla con columna `vector`.
- **Soporte de PDF** para manuales reales (`pypdf`).
- **Citas precisas:** que la respuesta señale documento y fragmento exacto.
- **Evals:** un set de preguntas con respuesta esperada para medir si la
  recuperación trae los chunks correctos (justo el "eval design" que el puesto
  lista como "nice to have").
- **Caché y control de latencia** en las llamadas a la API.