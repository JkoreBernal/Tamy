import json
import os
import re
from datetime import datetime

from core.motor_web import consultar_web


MEMORIA_CONOCIMIENTO_FILE = "data/conocimiento_tamy.json"

# Longitud máxima de una definición almacenada.
MAX_DEFINICION = 300


# ============================================================
# CARGA Y GUARDADO
# ============================================================

def cargar_base_conocimiento():
    """Carga la base de conocimiento de TAMY."""

    if os.path.exists(MEMORIA_CONOCIMIENTO_FILE):

        try:

            with open(
                MEMORIA_CONOCIMIENTO_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                datos = json.load(f)

            # Reparar estructuras incompletas
            if not isinstance(datos, dict):
                datos = {}

            datos.setdefault("definiciones", {})
            datos.setdefault("metadatos", {})

            return datos

        except Exception as error:

            print(
                f"[MotorConocimiento] "
                f"No se pudo cargar la memoria: {error}"
            )

    return {
        "definiciones": {
            "ford": (
                "Empresa estadounidense dedicada "
                "a la fabricación de automóviles."
            )
        },

        "metadatos": {}
    }


def guardar_base_conocimiento(datos):
    """Guarda la base de conocimiento."""

    carpeta = os.path.dirname(
        MEMORIA_CONOCIMIENTO_FILE
    )

    if carpeta:
        os.makedirs(
            carpeta,
            exist_ok=True
        )

    with open(
        MEMORIA_CONOCIMIENTO_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            datos,
            f,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# LIMPIEZA DE TEXTO
# ============================================================

def limpiar_texto(texto):
    """
    Limpia espacios, saltos de línea y basura
    innecesaria procedente de resultados web.
    """

    if not texto:
        return ""

    texto = str(texto)

    # Eliminar saltos de línea repetidos
    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    # Eliminar espacios antes/después
    texto = texto.strip()

    return texto


def eliminar_basura_web(texto):
    """
    Elimina fragmentos típicos de anuncios,
    navegación y contenido irrelevante.
    """

    if not texto:
        return ""

    basura = [
        "ad viewing ads",
        "privacy protected by duckduckgo",
        "clicks are managed by microsoft",
        "cookie",
        "cookies",
        "sign in",
        "log in",
        "suscríbete",
        "suscribete",
        "iniciar sesión",
        "iniciar sesion",
        "ver más",
        "ver mas",
    ]

    partes = texto.split(".")

    limpias = []

    for parte in partes:

        parte_limpia = parte.strip()

        if not parte_limpia:
            continue

        parte_normalizada = parte_limpia.lower()

        if any(
            palabra in parte_normalizada
            for palabra in basura
        ):
            continue

        limpias.append(
            parte_limpia
        )

    return ". ".join(limpias).strip()


def limitar_definicion(texto):
    """
    Convierte un texto largo en una definición corta.
    Intenta conservar una o dos frases completas.
    """

    texto = limpiar_texto(texto)

    if not texto:
        return ""

    if len(texto) <= MAX_DEFINICION:
        return texto

    # Buscar el primer punto después de un tamaño razonable
    corte = texto.find(".", 80)

    if corte != -1 and corte <= MAX_DEFINICION:
        return texto[:corte + 1].strip()

    # Si no encontramos una frase adecuada,
    # cortar limpiamente.
    texto = texto[:MAX_DEFINICION]

    ultimo_espacio = texto.rfind(" ")

    if ultimo_espacio > 100:
        texto = texto[:ultimo_espacio]

    return texto.rstrip(" ,;:") + "."


# ============================================================
# EXTRACCIÓN DE DEFINICIONES
# ============================================================

def extraer_definicion(concepto, informacion):
    """
    Recibe los resultados del motor web y trata de obtener
    únicamente una definición útil y corta.

    IMPORTANTE:
    Este proceso todavía NO utiliza una IA externa.
    Es un filtro heurístico local.
    """

    if not informacion:
        return None

    texto = limpiar_texto(
        informacion
    )

    texto = eliminar_basura_web(
        texto
    )

    if not texto:
        return None

    concepto_limpio = concepto.lower().strip()

    # --------------------------------------------------------
    # Buscar frases que parezcan definiciones
    # --------------------------------------------------------

    patrones = [
        rf"{re.escape(concepto_limpio)}\s+es\s+[^.]+",
        rf"{re.escape(concepto_limpio)}\s+es\s+un[ao]?\s+[^.]+",
        rf"{re.escape(concepto_limpio)}\s+se\s+define\s+como\s+[^.]+",
        rf"{re.escape(concepto_limpio)}\s+consiste\s+en\s+[^.]+",
    ]

    for patron in patrones:

        coincidencia = re.search(
            patron,
            texto,
            re.IGNORECASE
        )

        if coincidencia:

            definicion = coincidencia.group(0)

            return limitar_definicion(
                definicion
            )

    # --------------------------------------------------------
    # Buscar la primera frase útil
    # --------------------------------------------------------

    frases = re.split(
        r"(?<=[.!?])\s+",
        texto
    )

    for frase in frases:

        frase = frase.strip()

        if len(frase) < 30:
            continue

        if len(frase) > 500:
            continue

        return limitar_definicion(
            frase
        )

    # --------------------------------------------------------
    # Último recurso
    # --------------------------------------------------------

    return limitar_definicion(
        texto
    )


# ============================================================
# APRENDIZAJE
# ============================================================

def aprender_concepto(
    clave,
    definicion,
    fuente="web"
):
    """
    Guarda una definición limpia en la memoria
    de conocimiento de TAMY.
    """

    clave = limpiar_texto(
        clave
    ).lower()

    definicion = limitar_definicion(
        definicion
    )

    if not clave or not definicion:
        return False

    base = cargar_base_conocimiento()

    base.setdefault(
        "definiciones",
        {}
    )

    base.setdefault(
        "metadatos",
        {}
    )

    base["definiciones"][clave] = definicion

    base["metadatos"][clave] = {
        "fuente": fuente,
        "fecha": datetime.now().isoformat(
            timespec="seconds"
        )
    }

    guardar_base_conocimiento(
        base
    )

    return True


# ============================================================
# MEMORIA LOCAL
# ============================================================

def buscar_conocimiento_local(concepto):
    """Busca una definición en la memoria interna."""

    base = cargar_base_conocimiento()

    concepto_clean = (
        concepto.lower()
        .strip()
    )

    if not concepto_clean:
        return None

    definiciones = base.get(
        "definiciones",
        {}
    )

    # Coincidencia exacta
    if concepto_clean in definiciones:

        return definiciones[
            concepto_clean
        ]

    # Coincidencia parcial
    for clave, definicion in definiciones.items():

        clave_normalizada = (
            clave.lower()
            .strip()
        )

        if (
            clave_normalizada in concepto_clean
            or concepto_clean in clave_normalizada
        ):

            return definicion

    return None


# ============================================================
# AUTODEFINICIÓN
# ============================================================

def crear_autodefinicion(concepto):
    """
    Crea un nodo provisional.

    Esto NO pretende inventar conocimiento.
    Simplemente registra que el concepto existe
    y necesita investigación posterior.
    """

    return (
        f"{concepto.capitalize()} es un concepto "
        f"que TAMY todavía necesita investigar."
    )


# ============================================================
# INVESTIGACIÓN COMPLETA
# ============================================================

def obtener_o_investigar_definicion(concepto):
    """
    Cerebro principal del motor de conocimiento.

    Flujo:

        CONCEPTO
           ↓
        MEMORIA
           ↓
        ¿Existe?
        /     \
      SI       NO
      ↓         ↓
   devolver   WEB
                ↓
          limpiar datos
                ↓
        extraer definición
                ↓
             guardar
                ↓
            devolver
    """

    concepto_clean = limpiar_texto(
        concepto
    ).lower()

    if not concepto_clean:

        return (
            "No recibí ningún concepto "
            "para investigar."
        )

    # ========================================================
    # 1. MEMORIA INTERNA
    # ========================================================

    definicion = buscar_conocimiento_local(
        concepto_clean
    )

    if definicion:

        return (
            f"🧠 Memoria interna:\n"
            f"{definicion}"
        )

    # ========================================================
    # 2. WEB
    # ========================================================

    print(
        f"🌐 [MotorConocimiento] "
        f"Buscando información sobre: "
        f"{concepto_clean}"
    )

    try:

        informacion_web = consultar_web(
            concepto_clean
        )

    except Exception as error:

        print(
            f"[MotorConocimiento] "
            f"Error web: {error}"
        )

        informacion_web = None

    # ========================================================
    # 3. PROCESAR RESULTADO WEB
    # ========================================================

    if informacion_web:

        definicion = extraer_definicion(
            concepto_clean,
            informacion_web
        )

        if definicion:

            aprender_concepto(
                concepto_clean,
                definicion,
                fuente="web"
            )

            return (
                f"🌐 Investigación web realizada.\n\n"
                f"📖 {definicion}\n\n"
                f"🧠 He guardado una definición "
                f"corta en mi base de conocimiento."
            )

    # ========================================================
    # 4. AUTODEFINICIÓN PROVISIONAL
    # ========================================================

    definicion_autogenerada = (
        crear_autodefinicion(
            concepto_clean
        )
    )

    aprender_concepto(
        concepto_clean,
        definicion_autogenerada,
        fuente="provisional"
    )

    return (
        f"💡 No encontré información suficiente "
        f"sobre '{concepto_clean}'.\n\n"
        f"🧠 He registrado el concepto como "
        f"conocimiento provisional."
    )


# ============================================================
# INFORMACIÓN DEL MOTOR
# ============================================================

def estadisticas_conocimiento():
    """
    Devuelve estadísticas simples de la memoria.
    """

    base = cargar_base_conocimiento()

    definiciones = base.get(
        "definiciones",
        {}
    )

    return {
        "conceptos": len(definiciones),
        "archivo": MEMORIA_CONOCIMIENTO_FILE
    }


def listar_conocimientos():
    """Devuelve todos los conceptos conocidos."""

    base = cargar_base_conocimiento()

    return list(
        base.get(
            "definiciones",
            {}
        ).keys()
    )


# ============================================================
# PRUEBA
# ============================================================

if __name__ == "__main__":

    print(
        obtener_o_investigar_definicion(
            "Ferrari"
        )
    )

    print()

    print(
        estadisticas_conocimiento()
    )