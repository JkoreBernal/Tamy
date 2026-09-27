def guardar_contexto(memoria, clave, valor):
    memoria["contexto"][clave] = valor


def obtener_contexto(memoria, clave):
    return memoria["contexto"].get(clave)