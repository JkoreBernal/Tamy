from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher, get_close_matches
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ================================================================
# TAMY - MOTOR DE HIPÓTESIS v2
# ================================================================
# Objetivos:
#   1. Detectar intención sin depender de una frase exacta.
#   2. Soportar sinónimos y variantes comunes.
#   3. Tolerar errores de escritura leves.
#   4. Generar varias hipótesis ordenadas por confianza.
#   5. Mantener la API original:
#          analizar(mensaje)
#          elegir_pregunta(resultados)
#   6. Ser fácil de ampliar desde un solo archivo.
#
# Este módulo NO ejecuta acciones del sistema. Solamente interpreta.
# ================================================================


@dataclass(frozen=True)
class IntentConfig:
    """Definición de una intención."""

    nombre: str
    palabras: Tuple[str, ...] = ()
    frases: Tuple[str, ...] = ()
    sinonimos: Tuple[str, ...] = ()
    preguntas: Tuple[str, ...] = ()
    categoria: str = "general"
    peso_frase: float = 6.0
    peso_palabra: float = 2.0
    peso_sinonimo: float = 2.5
    requiere_confirmacion: bool = False

    def pregunta(self) -> str:
        if self.preguntas:
            return self.preguntas[0]
        return f"¿Te refieres a '{self.nombre}'?"


@dataclass
class Hipotesis:
    """Resultado de una hipótesis candidata."""

    intencion: str
    puntuacion: float
    pregunta: str
    categoria: str
    evidencias: List[str] = field(default_factory=list)
    requiere_confirmacion: bool = False

    def to_dict(self) -> dict:
        return {
            "intencion": self.intencion,
            "puntuacion": round(self.puntuacion, 3),
            "pregunta": self.pregunta,
            "categoria": self.categoria,
            "evidencias": list(self.evidencias),
            "requiere_confirmacion": self.requiere_confirmacion,
        }


# ================================================================
# NORMALIZACIÓN
# ================================================================


def quitar_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto)
    return "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )


def normalizar(texto: str) -> str:
    """Normaliza texto para comparación aproximada."""
    texto = str(texto or "")
    texto = texto.lower().strip()
    texto = quitar_acentos(texto)
    texto = texto.replace("¿", " ").replace("¡", " ")
    texto = re.sub(r"[^a-z0-9_áéíóúüñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def tokens(texto: str) -> List[str]:
    return normalizar(texto).split()


def token_set(texto: str) -> set[str]:
    return set(tokens(texto))


def similitud(a: str, b: str) -> float:
    a = normalizar(a)
    b = normalizar(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def contiene_frase(mensaje: str, frase: str) -> bool:
    return normalizar(frase) in normalizar(mensaje)


def distancia_textual(mensaje: str, termino: str) -> float:
    """Similitud entre un término y cualquiera de los tokens del mensaje."""
    palabra = normalizar(termino)
    if not palabra:
        return 0.0
    mejor = 0.0
    for token in tokens(mensaje):
        mejor = max(mejor, similitud(token, palabra))
    return mejor


# ================================================================
# DICCIONARIO DE INTENCIONES
# ================================================================
# Nota: muchas variantes deliberadamente se solapan.
# Eso permite que el motor pueda producir hipótesis 1/2/3.
# ================================================================


def _i(
    nombre: str,
    palabras: Sequence[str] = (),
    frases: Sequence[str] = (),
    sinonimos: Sequence[str] = (),
    preguntas: Sequence[str] = (),
    categoria: str = "general",
    peso_frase: float = 6.0,
    peso_palabra: float = 2.0,
    peso_sinonimo: float = 2.5,
    requiere_confirmacion: bool = False,
) -> IntentConfig:
    return IntentConfig(
        nombre=nombre,
        palabras=tuple(palabras),
        frases=tuple(frases),
        sinonimos=tuple(sinonimos),
        preguntas=tuple(preguntas),
        categoria=categoria,
        peso_frase=peso_frase,
        peso_palabra=peso_palabra,
        peso_sinonimo=peso_sinonimo,
        requiere_confirmacion=requiere_confirmacion,
    )


INTENCIONES: Dict[str, IntentConfig] = {
    # ---------------- CONVERSACIÓN ----------------
    "saludar": _i(
        "saludar",
        palabras=("hola", "buenas", "hey", "saludos", "holi", "holaa"),
        frases=("que tal", "cómo estás", "como estas", "buen día", "buenas tardes", "buenas noches"),
        sinonimos=("saludar", "saludo"),
        preguntas=("¿Me estás saludando?",),
        categoria="conversacion",
    ),
    "despedida": _i(
        "despedida",
        palabras=("adios", "adiós", "chao", "bye", "luego"),
        frases=("hasta luego", "nos vemos", "hasta mañana"),
        sinonimos=("despedirse",),
        preguntas=("¿Te estás despidiendo?",),
        categoria="conversacion",
    ),
    "agradecimiento": _i(
        "agradecimiento",
        palabras=("gracias", "agradezco"),
        frases=("muchas gracias", "te agradezco", "gracias por ayudarme"),
        sinonimos=("agradecer",),
        preguntas=("¿Me estás agradeciendo?",),
        categoria="conversacion",
    ),
    "disculpa": _i(
        "disculpa",
        palabras=("perdón", "perdon", "disculpa", "disculpas"),
        frases=("lo siento", "una disculpa", "perdón por eso"),
        sinonimos=("disculparse",),
        preguntas=("¿Te estás disculpando?",),
        categoria="conversacion",
    ),
    "confirmacion": _i(
        "confirmacion",
        palabras=("si", "sí", "claro", "exacto", "correcto", "afirmativo"),
        frases=("así es", "eso es", "es correcto"),
        sinonimos=("afirmar", "confirmar"),
        preguntas=("¿Estás confirmando lo anterior?",),
        categoria="conversacion",
    ),
    "negacion": _i(
        "negacion",
        palabras=("no", "nop", "negativo", "nunca"),
        frases=("no es así", "para nada", "eso no"),
        sinonimos=("negar", "rechazar"),
        preguntas=("¿Estás rechazando la hipótesis anterior?",),
        categoria="conversacion",
    ),
    "broma": _i(
        "broma",
        palabras=("broma", "bromear", "jaja", "jajaja"),
        frases=("era broma", "estoy bromeando", "es broma"),
        sinonimos=("humor", "chiste"),
        preguntas=("¿Estás bromeando?",),
        categoria="conversacion",
    ),
    "queja": _i(
        "queja",
        palabras=("molesto", "molesta", "enojo", "enojado", "problema"),
        frases=("esto no funciona", "algo salió mal", "algo salio mal"),
        sinonimos=("quejarse", "frustracion", "frustración"),
        preguntas=("¿Quieres contarme cuál es el problema?",),
        categoria="conversacion",
    ),

    # ---------------- IDENTIDAD / AYUDA ----------------
    "identidad": _i(
        "identidad",
        palabras=("quien", "quién", "eres", "tamy", "nombre"),
        frases=("quien eres", "cómo te llamas", "como te llamas", "que eres", "qué eres"),
        sinonimos=("identidad",),
        preguntas=("¿Quieres saber quién soy?",),
        categoria="identidad",
    ),
    "capacidades": _i(
        "capacidades",
        palabras=("opciones", "capacidades", "funciones", "puedes", "hacer"),
        frases=("que puedes hacer", "qué puedes hacer", "que sabes hacer", "qué sabes hacer", "cuales son tus funciones", "cuáles son tus funciones"),
        sinonimos=("habilidades", "posibilidades", "caracteristicas", "características"),
        preguntas=("¿Quieres conocer mis capacidades?",),
        categoria="identidad",
    ),
    "ayuda": _i(
        "ayuda",
        palabras=("ayuda", "ayudame", "ayúdame", "help", "auxilio"),
        frases=("necesito ayuda", "puedes ayudarme", "me ayudas", "échame la mano", "echame la mano"),
        sinonimos=("asistencia", "apoyo", "socorro"),
        preguntas=("¿Necesitas que te muestre en qué puedo ayudarte?",),
        categoria="ayuda",
    ),
    "estado": _i(
        "estado",
        palabras=("estado", "online", "offline", "funcionando"),
        frases=("como estas", "cómo estás", "estas bien", "estás bien"),
        sinonimos=("estatus", "situacion", "situación"),
        preguntas=("¿Quieres conocer mi estado actual?",),
        categoria="identidad",
    ),

    # ---------------- TIEMPO ----------------
    "hora": _i(
        "hora",
        palabras=("hora", "reloj"),
        frases=("que hora es", "qué hora es", "que hora tienes", "qué hora tienes", "hora actual"),
        sinonimos=("tiempo",),
        preguntas=("¿Quieres saber la hora actual?",),
        categoria="tiempo",
    ),
    "fecha": _i(
        "fecha",
        palabras=("fecha", "dia", "día", "hoy"),
        frases=("que fecha es", "qué fecha es", "que día es", "qué día es", "fecha actual", "hoy que fecha"),
        sinonimos=("calendario",),
        preguntas=("¿Quieres saber la fecha actual?",),
        categoria="tiempo",
    ),
    "mes": _i(
        "mes",
        palabras=("mes", "mensual"),
        frases=("en que mes estamos", "en qué mes estamos", "que mes es", "qué mes es", "mes actual"),
        sinonimos=("mensualidad",),
        preguntas=("¿Quieres saber en qué mes estamos?",),
        categoria="tiempo",
    ),
    "año": _i(
        "año",
        palabras=("año", "ano", "anual"),
        frases=("en que año estamos", "en qué año estamos", "que año es", "qué año es", "año actual"),
        sinonimos=("anual",),
        preguntas=("¿Quieres saber en qué año estamos?",),
        categoria="tiempo",
    ),
    "dia_semana": _i(
        "dia_semana",
        palabras=("lunes", "martes", "miércoles", "miercoles", "jueves", "viernes", "sábado", "sabado", "domingo"),
        frases=("que dia cae", "qué día cae", "que día de la semana", "qué día de la semana"),
        sinonimos=("semana",),
        preguntas=("¿Quieres saber el día de la semana?",),
        categoria="tiempo",
    ),

    # ---------------- MEMORIA ----------------
    "memoria": _i(
        "memoria",
        palabras=("memoria", "recuerdas", "recuerdo", "recordar", "aprendido"),
        frases=("que recuerdas", "qué recuerdas", "que has aprendido", "qué has aprendido", "que recuerdas de mi", "qué recuerdas de mí"),
        sinonimos=("recordatorio", "historial"),
        preguntas=("¿Quieres consultar mi memoria?",),
        categoria="memoria",
    ),
    "gustos": _i(
        "gustos",
        palabras=("gusta", "gustos", "preferencias", "preferencia"),
        frases=("que te gusta", "qué te gusta", "cuales son tus gustos", "cuáles son tus gustos", "que prefieres", "qué prefieres"),
        sinonimos=("preferir", "preferencia", "aficion", "afición"),
        preguntas=("¿Quieres consultar mis preferencias?",),
        categoria="memoria",
    ),
    "guardar_memoria": _i(
        "guardar_memoria",
        palabras=("recuerda", "guarda", "memoriza", "anota"),
        frases=("recuerda esto", "guarda esto", "quiero que recuerdes", "anota esto"),
        sinonimos=("memorizar", "registrar", "almacenar"),
        preguntas=("¿Quieres que guarde esto en mi memoria?",),
        categoria="memoria",
        requiere_confirmacion=True,
    ),
    "borrar_memoria": _i(
        "borrar_memoria",
        palabras=("olvida", "borrar", "borra", "elimina", "eliminar"),
        frases=("olvida esto", "borra esto", "elimina este recuerdo", "quiero que olvides"),
        sinonimos=("olvidar", "eliminacion", "eliminación"),
        preguntas=("¿Quieres que elimine ese recuerdo?",),
        categoria="memoria",
        requiere_confirmacion=True,
    ),

    # ---------------- APRENDIZAJE ----------------
    "aprendizaje": _i(
        "aprendizaje",
        palabras=("pensar", "piensa", "aprender", "aprende", "enseñar", "enseña", "razonar", "razona", "entender", "entiende", "inteligencia", "cerebro", "aprenderas", "aprenderás"),
        frases=("hacerte pensar", "quiero que pienses", "quiero enseñarte", "quiero que aprendas", "quiero que razones", "quiero que entiendas", "me gustaría que aprendieras", "hacerte aprender"),
        sinonimos=("aprendizaje", "razonamiento", "cognicion", "cognición", "inteligencia", "entrenamiento"),
        preguntas=("¿Quieres que aprenda o mejore mi capacidad de interpretar lo que dices?",),
        categoria="aprendizaje",
        requiere_confirmacion=True,
    ),
    "enseñar": _i(
        "enseñar",
        palabras=("enseñar", "enseña", "explicar", "explicame", "explícame"),
        frases=("quiero enseñarte", "te voy a enseñar", "voy a enseñarte", "aprende esto"),
        sinonimos=("instruir", "capacitar", "formar"),
        preguntas=("¿Quieres enseñarme algo nuevo?",),
        categoria="aprendizaje",
        requiere_confirmacion=True,
    ),
    "razonamiento": _i(
        "razonamiento",
        palabras=("razonamiento", "razona", "razonar", "logica", "lógica"),
        frases=("quiero que razones", "haz el razonamiento", "piensa esto", "analiza esto"),
        sinonimos=("deduccion", "deducción", "inferir", "inferencia"),
        preguntas=("¿Quieres que analice el problema paso a paso?",),
        categoria="aprendizaje",
        requiere_confirmacion=True,
    ),

    # ---------------- SISTEMA ----------------
    "abrir_aplicacion": _i(
        "abrir_aplicacion",
        palabras=("abrir", "abre", "ejecutar", "ejecuta", "iniciar", "inicia", "programa", "aplicacion", "aplicación"),
        frases=("abre una aplicación", "abre una aplicacion", "quiero abrir un programa", "ejecuta un programa"),
        sinonimos=("lanzar", "iniciar", "arrancar"),
        preguntas=("¿Quieres abrir una aplicación?",),
        categoria="sistema",
        requiere_confirmacion=True,
    ),
    "cerrar_aplicacion": _i(
        "cerrar_aplicacion",
        palabras=("cerrar", "cierra", "terminar", "detener"),
        frases=("cierra una aplicación", "cierra el programa", "quiero cerrar un programa"),
        sinonimos=("finalizar", "salir", "apagar"),
        preguntas=("¿Quieres cerrar una aplicación?",),
        categoria="sistema",
        requiere_confirmacion=True,
    ),
    "abrir_archivo": _i(
        "abrir_archivo",
        palabras=("archivo", "documento", "pdf", "imagen", "foto", "carpeta", "directorio"),
        frases=("abrir archivo", "abrir documento", "buscar archivo", "abrir carpeta"),
        sinonimos=("fichero", "documento", "ruta"),
        preguntas=("¿Quieres trabajar con un archivo o documento?",),
        categoria="archivos",
        requiere_confirmacion=True,
    ),
    "buscar_archivo": _i(
        "buscar_archivo",
        palabras=("buscar", "busca", "encontrar", "encuentra", "localizar", "archivo"),
        frases=("quiero buscar un archivo", "donde está", "dónde está", "encuentra mi archivo", "busca un documento"),
        sinonimos=("localizar", "rastrear", "ubicar"),
        preguntas=("¿Quieres que busque un archivo?",),
        categoria="archivos",
        requiere_confirmacion=True,
    ),
    "crear_archivo": _i(
        "crear_archivo",
        palabras=("crear", "crea", "nuevo", "archivo", "documento"),
        frases=("crear un archivo", "crea un documento", "nuevo archivo", "hazme un archivo"),
        sinonimos=("generar", "producir", "hacer"),
        preguntas=("¿Quieres que cree un archivo?",),
        categoria="archivos",
        requiere_confirmacion=True,
    ),
    "listar_archivos": _i(
        "listar_archivos",
        palabras=("lista", "listar", "archivos", "carpeta", "directorio"),
        frases=("muestra los archivos", "lista los archivos", "que hay en esta carpeta", "qué hay en esta carpeta"),
        sinonimos=("mostrar", "contenido", "directorio"),
        preguntas=("¿Quieres que liste los archivos?",),
        categoria="archivos",
    ),
    "buscar": _i(
        "buscar",
        palabras=("buscar", "busca", "encontrar", "encuentra", "localizar"),
        frases=("quiero buscar", "ayudame a buscar", "ayúdame a buscar", "necesito encontrar"),
        sinonimos=("consulta", "investigar", "hallar"),
        preguntas=("¿Quieres que busque algo?",),
        categoria="busqueda",
    ),

    # ---------------- UNIVERSIDAD / PRODUCTIVIDAD ----------------
    "tareas": _i(
        "tareas",
        palabras=("tarea", "tareas", "pendiente", "pendientes", "trabajo", "trabajos", "universidad"),
        frases=("que tengo pendiente", "qué tengo pendiente", "que tengo que hacer", "qué tengo que hacer", "que tareas tengo", "qué tareas tengo"),
        sinonimos=("deberes", "asignaciones", "actividad", "actividades"),
        preguntas=("¿Estás preguntando por tus tareas o pendientes?",),
        categoria="universidad",
    ),
    "estudiar": _i(
        "estudiar",
        palabras=("estudiar", "estudio", "estudia", "repasar", "repaso"),
        frases=("quiero estudiar", "vamos a estudiar", "hora de estudiar", "necesito estudiar"),
        sinonimos=("repasar", "practicar", "preparar"),
        preguntas=("¿Quieres organizar una sesión de estudio?",),
        categoria="universidad",
    ),
    "trabajar": _i(
        "trabajar",
        palabras=("trabajar", "trabajo", "trabajando", "laborar", "laburar"),
        frases=("quiero trabajar", "necesito trabajar", "vamos a trabajar", "ponte a trabajar"),
        sinonimos=("laborar", "ocupacion", "ocupación", "produccion", "producción"),
        preguntas=("¿Quieres que te ayude a organizar lo que vas a trabajar?",),
        categoria="productividad",
    ),
    "proyecto": _i(
        "proyecto",
        palabras=("proyecto", "proyectos", "prototipo"),
        frases=("trabajar en el proyecto", "mi proyecto", "el proyecto"),
        sinonimos=("desarrollo", "sistema", "aplicacion", "aplicación"),
        preguntas=("¿Quieres trabajar en un proyecto?",),
        categoria="productividad",
    ),
    "recordatorio": _i(
        "recordatorio",
        palabras=("recordatorio", "recordar", "avísame", "avisame", "alerta", "alarma"),
        frases=("recuérdame", "recuerdame", "avísame luego", "avisame luego", "ponme un recordatorio"),
        sinonimos=("notificacion", "notificación", "alarma"),
        preguntas=("¿Quieres crear un recordatorio?",),
        categoria="productividad",
        requiere_confirmacion=True,
    ),

    # ---------------- INFORMACIÓN ----------------
    "explicar": _i(
        "explicar",
        palabras=("explica", "explicar", "explicame", "explícame", "entender"),
        frases=("puedes explicar", "me puedes explicar", "explícame esto", "explicame esto"),
        sinonimos=("aclarar", "describir", "enseñar"),
        preguntas=("¿Quieres que te explique algo?",),
        categoria="informacion",
    ),
    "definir": _i(
        "definir",
        palabras=("define", "definir", "significa", "significado"),
        frases=("que significa", "qué significa", "que es", "qué es"),
        sinonimos=("concepto", "definicion", "definición"),
        preguntas=("¿Quieres que defina un concepto?",),
        categoria="informacion",
    ),
    "resumir": _i(
        "resumir",
        palabras=("resumen", "resumir", "resume"),
        frases=("hazme un resumen", "puedes resumir", "resume esto"),
        sinonimos=("sintesis", "síntesis", "extracto"),
        preguntas=("¿Quieres que haga un resumen?",),
        categoria="informacion",
    ),
    "comparar": _i(
        "comparar",
        palabras=("comparar", "compara", "diferencia", "diferencias"),
        frases=("cual es la diferencia", "cuál es la diferencia", "compara esto", "que diferencia hay", "qué diferencia hay"),
        sinonimos=("contrastar", "versus", "vs"),
        preguntas=("¿Quieres que compare dos cosas?",),
        categoria="informacion",
    ),
    "calcular": _i(
        "calcular",
        palabras=("calcular", "calcula", "cuenta", "suma", "resta", "multiplica", "divide"),
        frases=("haz la cuenta", "cuánto es", "cuanto es", "resuelve esto"),
        sinonimos=("operacion", "operación", "aritmetica", "aritmética"),
        preguntas=("¿Quieres que haga un cálculo?",),
        categoria="informacion",
    ),

    # ---------------- EMOCIÓN / INTERACCIÓN ----------------
    "opinion": _i(
        "opinion",
        palabras=("opina", "opinion", "opinión", "piensas", "crees"),
        frases=("que opinas", "qué opinas", "que piensas", "qué piensas", "cual crees", "cuál crees"),
        sinonimos=("perspectiva", "valoracion", "valoración"),
        preguntas=("¿Quieres que te dé una perspectiva sobre eso?",),
        categoria="conversacion",
    ),
    "curiosidad": _i(
        "curiosidad",
        palabras=("curioso", "curiosidad", "interesante", "pregunta"),
        frases=("tengo una pregunta", "te quiero preguntar", "quiero saber"),
        sinonimos=("duda", "cuestion", "cuestión"),
        preguntas=("¿Quieres hacerme una pregunta?",),
        categoria="conversacion",
    ),
}


# ================================================================
# EXPANSIONES Y CORRECCIONES COMUNES
# ================================================================

CORRECCIONES = {
    "tranbajar": "trabajar",
    "travajar": "trabajar",
    "trabajar": "trabajar",
    "aprenderas": "aprenderás",
    "q": "que",
    "k": "que",
    "kien": "quien",
    "ke": "que",
    "tamy": "tamy",
    "aplicacion": "aplicación",
    "aplicacion": "aplicación",
    "memria": "memoria",
    "recodar": "recordar",
    "horra": "hora",
    "fehca": "fecha",
    "mes": "mes",
}


PALABRAS_FUNCIONALES = {
    "quiero",
    "quieres",
    "necesito",
    "necesitamos",
    "me",
    "te",
    "que",
    "un",
    "una",
    "el",
    "la",
    "los",
    "las",
    "algo",
    "esto",
    "eso",
    "hacer",
    "con",
    "para",
    "de",
    "del",
    "por",
    "mi",
    "tu",
    "tú",
    "yo",
    "ahora",
    "hoy",
}


def corregir_token(token: str) -> str:
    token_normalizado = normalizar(token)

    if token_normalizado in CORRECCIONES:
        return normalizar(CORRECCIONES[token_normalizado])

    return token_normalizado


def corregir_tokens(mensaje: str) -> List[str]:
    return [corregir_token(token) for token in tokens(mensaje)]


def _vocabulario() -> List[str]:
    vocabulario: set[str] = set()

    for datos in INTENCIONES.values():
        vocabulario.update(normalizar(x) for x in datos.palabras)
        vocabulario.update(normalizar(x) for x in datos.sinonimos)

    return sorted(vocabulario)


VOCABULARIO = _vocabulario()


# ================================================================
# CÁLCULO DE PUNTUACIONES
# ================================================================


def _puntuacion_frases(mensaje: str, datos: IntentConfig) -> Tuple[float, List[str]]:
    puntos = 0.0
    evidencias: List[str] = []

    for frase in datos.frases:
        if contiene_frase(mensaje, frase):
            puntos += datos.peso_frase
            evidencias.append(f"frase:{frase}")

    return puntos, evidencias


def _puntuacion_palabras(mensaje: str, datos: IntentConfig) -> Tuple[float, List[str]]:
    mensaje_tokens = set(corregir_tokens(mensaje))
    puntos = 0.0
    evidencias: List[str] = []

    for palabra in datos.palabras:
        candidata = normalizar(palabra)
        if candidata in mensaje_tokens:
            puntos += datos.peso_palabra
            evidencias.append(f"palabra:{candidata}")

    return puntos, evidencias


def _puntuacion_sinonimos(mensaje: str, datos: IntentConfig) -> Tuple[float, List[str]]:
    puntos = 0.0
    evidencias: List[str] = []
    mensaje_tokens = set(corregir_tokens(mensaje))

    for sinonimo in datos.sinonimos:
        candidato = normalizar(sinonimo)
        if candidato in mensaje_tokens:
            puntos += datos.peso_sinonimo
            evidencias.append(f"sinonimo:{candidato}")

    return puntos, evidencias


def _puntuacion_aproximada(mensaje: str, datos: IntentConfig) -> Tuple[float, List[str]]:
    puntos = 0.0
    evidencias: List[str] = []
    tokens_mensaje = corregir_tokens(mensaje)

    objetivos = list(datos.palabras) + list(datos.sinonimos)

    for token in tokens_mensaje:
        if len(token) < 4:
            continue
        for objetivo in objetivos:
            objetivo_n = normalizar(objetivo)
            if len(objetivo_n) < 4:
                continue

            ratio = similitud(token, objetivo_n)

            if ratio >= 0.86 and token != objetivo_n:
                puntos += 1.25 * ratio
                evidencias.append(
                    f"aproximado:{token}~{objetivo_n}:{ratio:.2f}"
                )
                break

    return puntos, evidencias


def _penalizacion_funcional(mensaje: str, datos: IntentConfig) -> float:
    """Penaliza coincidencias débiles causadas por palabras genéricas."""
    mensaje_set = set(tokens(mensaje))
    coincidencias = len(mensaje_set.intersection(PALABRAS_FUNCIONALES))

    if coincidencias >= 5 and not datos.frases:
        return 0.5

    return 0.0


def puntuar_intencion(mensaje: str, datos: IntentConfig) -> Tuple[float, List[str]]:
    total = 0.0
    evidencias: List[str] = []

    puntos, ev = _puntuacion_frases(mensaje, datos)
    total += puntos
    evidencias.extend(ev)

    puntos, ev = _puntuacion_palabras(mensaje, datos)
    total += puntos
    evidencias.extend(ev)

    puntos, ev = _puntuacion_sinonimos(mensaje, datos)
    total += puntos
    evidencias.extend(ev)

    puntos, ev = _puntuacion_aproximada(mensaje, datos)
    total += puntos
    evidencias.extend(ev)

    total -= _penalizacion_funcional(mensaje, datos)

    return max(total, 0.0), evidencias


# ================================================================
# REGLAS DE CONTEXTO
# ================================================================

CONTEXTO_PRIORIDADES = {
    "tareas": ("trabajar", "universidad", "proyecto", "recordatorio"),
    "aprendizaje": ("enseñar", "razonamiento", "explicar"),
    "archivos": ("abrir_archivo", "buscar_archivo", "crear_archivo"),
    "sistema": ("abrir_aplicacion", "cerrar_aplicacion"),
    "tiempo": ("hora", "fecha", "mes", "año", "dia_semana"),
}


def ajustar_por_contexto(
    resultados: List[Hipotesis],
    contexto: Optional[dict],
) -> List[Hipotesis]:
    if not contexto:
        return resultados

    ultima_intencion = contexto.get("ultima_intencion")
    historial = contexto.get("historial_intenciones", [])

    if ultima_intencion:
        for resultado in resultados:
            if resultado.intencion == ultima_intencion:
                resultado.puntuacion += 0.75
                resultado.evidencias.append("contexto:ultima_intencion")

    for resultado in resultados:
        if resultado.intencion in historial:
            resultado.puntuacion += 0.15
            resultado.evidencias.append("contexto:historial")

    resultados.sort(key=lambda x: x.puntuacion, reverse=True)
    return resultados


# ================================================================
# ANÁLISIS PRINCIPAL
# ================================================================


def analizar(
    mensaje: str,
    contexto: Optional[dict] = None,
    max_resultados: int = 3,
    minimo: float = 0.75,
) -> List[dict]:
    """
    Analiza un mensaje y devuelve hasta 3 hipótesis.

    Cada resultado contiene:
        intencion
        puntuacion
        pregunta
        categoria
        evidencias
        requiere_confirmacion
    """
    mensaje_n = normalizar(mensaje)

    if not mensaje_n:
        return []

    resultados: List[Hipotesis] = []

    for nombre, datos in INTENCIONES.items():
        puntuacion, evidencias = puntuar_intencion(
            mensaje_n,
            datos,
        )

        if puntuacion >= minimo:
            resultados.append(
                Hipotesis(
                    intencion=nombre,
                    puntuacion=puntuacion,
                    pregunta=datos.pregunta(),
                    categoria=datos.categoria,
                    evidencias=evidencias,
                    requiere_confirmacion=datos.requiere_confirmacion,
                )
            )

    resultados = ajustar_por_contexto(
        resultados,
        contexto,
    )

    resultados.sort(
        key=lambda x: x.puntuacion,
        reverse=True,
    )

    # Evitar devolver dos entradas idénticas.
    vistos: set[str] = set()
    unicos: List[Hipotesis] = []

    for resultado in resultados:
        if resultado.intencion in vistos:
            continue
        vistos.add(resultado.intencion)
        unicos.append(resultado)

    return [
        resultado.to_dict()
        for resultado in unicos[:max_resultados]
    ]


# ================================================================
# PREGUNTA DE DESAMBIGUACIÓN
# ================================================================


def elegir_pregunta(resultados: Sequence[dict]) -> Optional[str]:
    if not resultados:
        return None

    return resultados[0].get("pregunta")


def segunda_hipotesis(resultados: Sequence[dict]) -> Optional[dict]:
    if len(resultados) < 2:
        return None
    return resultados[1]


def tercera_hipotesis(resultados: Sequence[dict]) -> Optional[dict]:
    if len(resultados) < 3:
        return None
    return resultados[2]


def diferencia_entre_mejores(resultados: Sequence[dict]) -> float:
    if len(resultados) < 2:
        return float("inf")

    return (
        float(resultados[0]["puntuacion"])
        - float(resultados[1]["puntuacion"])
    )


def es_ambigua(
    resultados: Sequence[dict],
    margen: float = 2.0,
) -> bool:
    if len(resultados) < 2:
        return False

    return diferencia_entre_mejores(resultados) <= margen


# ================================================================
# CONFIANZA
# ================================================================


def convertir_a_confianza(
    puntuacion: float,
    maximo_referencia: float = 12.0,
) -> float:
    """Convierte una puntuación heurística a 0..1."""
    if puntuacion <= 0:
        return 0.0

    if puntuacion >= maximo_referencia:
        return 1.0

    return round(puntuacion / maximo_referencia, 3)


def etiqueta_confianza(puntuacion: float) -> str:
    confianza = convertir_a_confianza(puntuacion)

    if confianza >= 0.85:
        return "muy alta"
    if confianza >= 0.65:
        return "alta"
    if confianza >= 0.45:
        return "media"
    if confianza >= 0.25:
        return "baja"
    return "muy baja"


# ================================================================
# UTILIDADES DE DEBUG
# ================================================================


def depurar(
    mensaje: str,
    contexto: Optional[dict] = None,
    max_resultados: int = 3,
) -> str:
    resultados = analizar(
        mensaje,
        contexto=contexto,
        max_resultados=max_resultados,
    )

    if not resultados:
        return "DEBUG - HIPÓTESIS: ninguna"

    lineas = ["DEBUG - HIPÓTESIS:"]

    for indice, resultado in enumerate(resultados, start=1):
        confianza = convertir_a_confianza(
            resultado["puntuacion"]
        )
        etiqueta = etiqueta_confianza(
            resultado["puntuacion"]
        )

        lineas.append(
            f"  {indice}. {resultado['intencion']} "
            f"-> {resultado['puntuacion']:.2f} puntos "
            f"({confianza:.0%}, {etiqueta})"
        )

        if resultado.get("evidencias"):
            for evidencia in resultado["evidencias"]:
                lineas.append(
                    f"      - {evidencia}"
                )

    return "\n".join(lineas)


# ================================================================
# CORRECCIÓN APROXIMADA DE PALABRAS
# ================================================================


def sugerir_correcciones(mensaje: str, limite: int = 5) -> List[Tuple[str, str, float]]:
    sugerencias: List[Tuple[str, str, float]] = []

    for token in corregir_tokens(mensaje):
        if len(token) < 4:
            continue

        candidatos = get_close_matches(
            token,
            VOCABULARIO,
            n=limite,
            cutoff=0.80,
        )

        for candidato in candidatos:
            ratio = similitud(token, candidato)
            if ratio >= 0.80 and token != candidato:
                sugerencias.append(
                    (token, candidato, round(ratio, 3))
                )

    # Quitar duplicados preservando orden.
    unicas = []
    vistos = set()

    for elemento in sugerencias:
        clave = elemento[:2]
        if clave in vistos:
            continue
        vistos.add(clave)
        unicas.append(elemento)

    return unicas


# ================================================================
# BÚSQUEDA DIRECTA DE INTENCIÓN
# ================================================================


def obtener_intencion(nombre: str) -> Optional[IntentConfig]:
    return INTENCIONES.get(normalizar(nombre))


def listar_intenciones(categoria: Optional[str] = None) -> List[str]:
    if categoria is None:
        return sorted(INTENCIONES.keys())

    return sorted(
        nombre
        for nombre, datos in INTENCIONES.items()
        if datos.categoria == categoria
    )


def listar_categorias() -> List[str]:
    return sorted(
        {
            datos.categoria
            for datos in INTENCIONES.values()
        }
    )


# ================================================================
# DETECTOR DE RESPUESTAS SÍ/NO
# ================================================================

AFIRMACIONES = {
    "si",
    "sí",
    "s",
    "claro",
    "correcto",
    "exacto",
    "afirmativo",
    "dale",
    "va",
    "ok",
    "okay",
    "confirmo",
}

NEGACIONES = {
    "no",
    "n",
    "nop",
    "negativo",
    "para nada",
    "nope",
}


def es_afirmacion(mensaje: str) -> bool:
    return normalizar(mensaje) in {
        normalizar(x) for x in AFIRMACIONES
    }


def es_negacion(mensaje: str) -> bool:
    return normalizar(mensaje) in {
        normalizar(x) for x in NEGACIONES
    }


# ================================================================
# PRUEBAS RÁPIDAS LOCALES
# ================================================================


def ejecutar_pruebas() -> None:
    casos = [
        "hola",
        "qué hora es",
        "que fecha es",
        "el mes",
        "quiero hacer algo",
        "quiero trabajar",
        "necesito ayuda",
        "quiero aprender algo",
        "me gustaria que aprendieras a entender mejor",
        "quiero tranbajar",
        "abre algo",
        "busca algo",
        "qué puedes hacer",
    ]

    for caso in casos:
        print("=" * 60)
        print(f"MENSAJE: {caso}")
        print(depurar(caso))
        print()


if __name__ == "__main__":
    ejecutar_pruebas()