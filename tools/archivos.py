import os


def listar_archivos(ruta="."):
    try:
        return os.listdir(ruta)
    except FileNotFoundError:
        return []