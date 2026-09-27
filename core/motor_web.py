# core/motor_web.py

import re
import html
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 "
    "Chrome/140.0 Safari/537.36"
)


def limpiar_texto(texto):
    """Limpia espacios y caracteres innecesarios."""

    texto = html.unescape(texto)

    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def buscar_duckduckgo(consulta):
    """
    Realiza una búsqueda utilizando DuckDuckGo.
    """

    url = (
        "https://html.duckduckgo.com/html/?q="
        + urllib.parse.quote_plus(consulta)
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT
        }
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=10
        ) as respuesta:

            contenido = respuesta.read()

    except Exception as error:

        print(
            f"[MotorWeb] Error de conexión: {error}"
        )

        return []


    soup = BeautifulSoup(
        contenido,
        "html.parser"
    )


    resultados = []


    for resultado in soup.select(
        ".result"
    )[:5]:

        titulo = resultado.select_one(
            ".result__title"
        )

        enlace = resultado.select_one(
            ".result__a"
        )

        descripcion = resultado.select_one(
            ".result__snippet"
        )


        if not titulo or not enlace:

            continue


        titulo_texto = limpiar_texto(
            titulo.get_text(" ", strip=True)
        )

        enlace_url = enlace.get(
            "href",
            ""
        )

        descripcion_texto = ""

        if descripcion:

            descripcion_texto = limpiar_texto(
                descripcion.get_text(
                    " ",
                    strip=True
                )
            )


        resultados.append({
            "titulo": titulo_texto,
            "url": enlace_url,
            "descripcion": descripcion_texto
        })


    return resultados


def seleccionar_informacion(resultados):
    """
    Convierte los resultados de búsqueda
    en información que TAMY pueda almacenar.
    """

    if not resultados:

        return None


    informacion = []


    for resultado in resultados:

        titulo = resultado.get(
            "titulo",
            ""
        )

        descripcion = resultado.get(
            "descripcion",
            ""
        )

        url = resultado.get(
            "url",
            ""
        )


        if descripcion:

            informacion.append(
                f"{titulo}: {descripcion}"
            )

        elif titulo:

            informacion.append(
                titulo
            )


    if not informacion:

        return None


    return "\n".join(
        informacion[:5]
    )


def consultar_web(concepto):
    """
    Motor principal de consulta web.

    Recibe un concepto o pregunta y devuelve
    información encontrada en Internet.
    """

    consulta = limpiar_texto(
        concepto
    )

    if not consulta:

        return None


    print(
        f"🌐 [MotorWeb] Investigando: {consulta}"
    )


    resultados = buscar_duckduckgo(
        consulta
    )


    informacion = seleccionar_informacion(
        resultados
    )


    if informacion:

        print(
            f"🌐 [MotorWeb] "
            f"Encontrados {len(resultados)} resultados."
        )

        return informacion


    print(
        "🌐 [MotorWeb] No se encontró información."
    )

    return None


def investigar(consulta):
    """
    Alias sencillo para utilizar el motor
    desde otros módulos.
    """

    return consultar_web(
        consulta
    )


if __name__ == "__main__":

    print(
        "\n================================"
    )

    print(
        "       MOTOR WEB DE TAMY"
    )

    print(
        "================================\n"
    )


    consulta = input(
        "¿Qué quieres investigar?: "
    )


    resultado = consultar_web(
        consulta
    )


    if resultado:

        print(
            "\n📚 INFORMACIÓN ENCONTRADA:\n"
        )

        print(
            resultado
        )

    else:

        print(
            "\n❌ No se encontró información."
        )