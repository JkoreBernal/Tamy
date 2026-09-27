import json
import os

RUTA_MEMORIA = "data/memoria.json"


def cargar_memoria():
    if not os.path.exists(RUTA_MEMORIA):
        return {
            "nombre": "TAMY",
            "version": "0.1",
            "interacciones": 0,
            "gustos": [],
            "contexto": {}
        }

    with open(RUTA_MEMORIA, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


def guardar_memoria(memoria):
    os.makedirs("data", exist_ok=True)

    with open(RUTA_MEMORIA, "w", encoding="utf-8") as archivo:
        json.dump(memoria, archivo, indent=4, ensure_ascii=False)


def agregar_gusto(gusto):
    memoria = cargar_memoria()

    if gusto not in memoria["gustos"]:
        memoria["gustos"].append(gusto)

    guardar_memoria(memoria)