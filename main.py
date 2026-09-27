from core.tamy_brain import procesar_texto


def main():
    print("================================")
    print("          TAMY v0.1")
    print("================================")
    print("Escribe 'salir' para terminar.\n")

    while True:

        mensaje = input("Tú: ")

        if mensaje.lower().strip() == "salir":
            print("TAMY: Hasta luego.")
            break

        respuesta = procesar_texto(
            mensaje
        )

        print(
            f"TAMY: {respuesta}"
        )


if __name__ == "__main__":
    main()