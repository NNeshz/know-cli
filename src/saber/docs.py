"""Cargar documentos del disco y partirlos en chunks.

Este módulo es el primer eslabón del pipeline de RAG:
    documentos → [chunks] → índice → recuperar → generar

Un *chunk* es un pedazo de texto. Partimos los documentos en pedazos chicos para
luego poder recuperar solo el fragmento relevante a una pregunta, en vez de
mandar el documento entero. El *solape* entre pedazos evita cortar una idea por
la mitad en las costuras.
"""

from dataclasses import dataclass
from pathlib import Path

# Extensiones de archivo que consideramos "documentos de texto".
EXTENSIONES = {".txt", ".md"}


@dataclass
class Chunk:
    """Un pedazo de un documento, con su origen y su posición.

    - texto: el contenido del pedazo.
    - fuente: el nombre del archivo del que salió (para poder citarlo).
    - indice: su posición dentro de ese archivo (0 = primer pedazo).
    """

    texto: str
    fuente: str
    indice: int


def partir_texto(texto: str, tam: int = 800, solape: int = 100) -> list[str]:
    """Parte un texto en pedazos de ~`tam` caracteres con `solape` de traslape.

    Es una función *pura*: no toca el disco ni la red, solo transforma texto en
    una lista de textos. Eso la hace fácil de probar (Fase 5).

    Avanza de a `tam - solape` caracteres, así cada pedazo repite los últimos
    `solape` caracteres del anterior.
    """
    if tam <= 0:
        raise ValueError(f"'tam' debe ser positivo; recibí {tam}.")
    if not 0 <= solape < tam:
        raise ValueError(
            f"'solape' debe estar en el rango [0, tam); recibí solape={solape}, tam={tam}."
        )

    pedazos: list[str] = []
    inicio = 0
    paso = tam - solape
    while inicio < len(texto):
        pedazos.append(texto[inicio : inicio + tam])
        inicio += paso
    return pedazos


def cargar_chunks(carpeta: Path, tam: int = 800, solape: int = 100) -> list[Chunk]:
    """Recorre los .txt/.md de `carpeta` y devuelve todos sus chunks.

    Esta función sí hace I/O (lee archivos). Reusa `partir_texto` para la lógica
    pura del corte, manteniendo separadas "leer del disco" y "partir el texto".
    """
    if not carpeta.is_dir():
        raise NotADirectoryError(
            f"No existe la carpeta o no es un directorio: {carpeta}"
        )

    chunks: list[Chunk] = []
    for archivo in sorted(carpeta.glob("**/*")):
        if archivo.suffix not in EXTENSIONES:
            continue
        texto = archivo.read_text(encoding="utf-8")
        for i, pedazo in enumerate(partir_texto(texto, tam, solape)):
            chunks.append(Chunk(texto=pedazo, fuente=archivo.name, indice=i))
    return chunks
