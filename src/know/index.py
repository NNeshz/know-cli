"""Persistencia del índice en disco como JSON.

Guardamos los chunks (el "corpus") en un archivo JSON. Las estructuras de TF-IDF
NO se guardan: se recalculan rápido al cargar, llamando a `retriever.indexar()`.
Así el archivo en disco es simple y legible, y el índice = los chunks.
"""

import json
from dataclasses import asdict
from pathlib import Path

from know.docs import Chunk

NOMBRE_ARCHIVO = "chunks.json"
NOMBRE_VECTORES = "vectores.json"  # solo existe en índices del modo semántico


def guardar_indice(
    chunks: list[Chunk],
    carpeta: Path,
    vectores: list[list[float]] | None = None,
) -> None:
    """Guarda los chunks (y, en modo semántico, sus vectores) dentro de `carpeta`."""
    carpeta.mkdir(parents=True, exist_ok=True)
    datos = [asdict(c) for c in chunks]  # cada Chunk -> dict {texto, fuente, indice}
    ruta = carpeta / NOMBRE_ARCHIVO
    with ruta.open("w", encoding="utf-8") as archivo:
        # ensure_ascii=False conserva acentos y ñ legibles en el JSON.
        json.dump(datos, archivo, ensure_ascii=False, indent=2)

    ruta_vectores = carpeta / NOMBRE_VECTORES
    if vectores is not None:
        with ruta_vectores.open("w", encoding="utf-8") as archivo:
            json.dump(vectores, archivo)
    elif ruta_vectores.exists():
        # Si reindexas en modo léxico, borramos vectores viejos para no mezclar.
        ruta_vectores.unlink()


def cargar_indice(carpeta: Path) -> list[Chunk]:
    """Carga los chunks desde el JSON guardado en `carpeta`."""
    ruta = carpeta / NOMBRE_ARCHIVO
    if not ruta.exists():
        raise FileNotFoundError(
            f"No encontré un índice en {ruta}. ¿Corriste 'know indexar' antes?"
        )
    with ruta.open(encoding="utf-8") as archivo:
        datos = json.load(archivo)
    return [
        Chunk(texto=d["texto"], fuente=d["fuente"], indice=d["indice"]) for d in datos
    ]


def cargar_vectores(carpeta: Path) -> list[list[float]] | None:
    """Carga los vectores del índice si existen (solo en modo semántico)."""
    ruta = carpeta / NOMBRE_VECTORES
    if not ruta.exists():
        return None
    with ruta.open(encoding="utf-8") as archivo:
        vectores: list[list[float]] = json.load(archivo)
    return vectores
