"""Punto de entrada para ejecutar el paquete con `python -m saber`.

El comando instalado `saber` (definido en pyproject.toml) es la vía normal.
Este archivo es la vía alternativa: correr el paquete directamente como módulo.
Ambos terminan llamando a la misma `app`.
"""

from saber.cli import app

if __name__ == "__main__":
    app()
