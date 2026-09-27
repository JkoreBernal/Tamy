# TAMY BRAIN v2.0
# Cerebro central de TAMY.
# Coordina: entrada, contexto, hipótesis, aprendizaje, memoria,
# confirmación, respuestas, herramientas futuras y diagnóstico.

from __future__ import annotations

import datetime
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.memoria import cargar_memoria, guardar_memoria, agregar_gusto
from core.personalidad import obtener_presentacion
from core.motor_hipotesis import analizar, elegir_pregunta
from core.motor_aprendizaje import obtener_motor


NOMBRE = "TAMY"
VERSION = "2.0"
MAX_HIPOTESIS = 3
UMBRAL_RESPUESTA_DIRECTA = 7.0
UMBRAL_CONFIRMACION = 1.5
RUTA_ESTADO = Path("data/tamy_brain_state.json")


# ================================================================
# ESTRUCTURAS CENTRALES
# ================================================================


@dataclass
class HipotesisActiva:
    intencion: str
    puntuacion: float
    pregunta: str = ""
    confianza: float = 0.0
    evidencias: List[str] = field(default_factory=list)
    posicion: int = 0


@dataclass
class Turno:
    usuario: str
    tamy: str
    intencion: str = ""
    confirmada: Optional[bool] = None
    timestamp: str = ""


# Estado de conversación compartido por la sesión actual.
estado_tamy: Dict[str, Any] = {
    "candidatas": [],
    "mensaje_original": "",
    "esperando_confirmacion": False,
    "turnos": [],
    "ultima_intencion": "",
    "ultimo_mensaje": "",
    "ultimo_resultado": None,
    "modo": "normal",
}


motor_aprendizaje = obtener_motor()


# ================================================================
# CONFIGURACIÓN DEL CEREBRO
# ================================================================


INTENCIONES_CONFIRMACION = {
    "aprendizaje",
    "trabajar",
    "crear_archivo",
    "abrir_aplicacion",
    "abrir_archivo",
    "buscar",
    "guardar_memoria",
    "proyecto",
    "curiosidad",
}

INTENCIONES_DIRECTAS = {
    "saludar",
    "despedida",
    "agradecimiento",
    "identidad",
    "hora",
    "fecha",
    "mes",
    "año",
    "memoria",
    "gustos",
    "capacidades",
    "ayuda",
}


# ================================================================
# UTILIDADES GENERALES
# ================================================================


def ahora() -> datetime.datetime:
    return datetime.datetime.now()


def ahora_texto() -> str:
    return ahora().strftime("%Y-%m-%d %H:%M:%S")


def normalizar(texto: str) -> str:
    texto = texto.lower().strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def limpiar_puntuacion(texto: str) -> str:
    return re.sub(r"[¿?!¡.,;:()]", "", texto)


def tokens(texto: str) -> List[str]:
    limpio = limpiar_puntuacion(normalizar(texto))
    return [x for x in limpio.split(" ") if x]


def es_vacio(texto: str) -> bool:
    return not bool(normalizar(texto))


def limitar_texto(texto: str, limite: int = 5000) -> str:
    texto = str(texto)
    return texto if len(texto) <= limite else texto[:limite] + "…"


def es_si(texto: str) -> bool:
    return normalizar(texto) in {"si", "sí", "s", "claro", "correcto", "exacto", "afirmativo", "asi es", "así es"}


def es_no(texto: str) -> bool:
    return normalizar(texto) in {"no", "n", "nop", "negativo"}


def es_comando(texto: str, *comandos: str) -> bool:
    valor = normalizar(texto)
    return valor in {normalizar(x) for x in comandos}


def formatear_intencion(nombre: str) -> str:
    return str(nombre).replace("_", " ").strip()


def porcentaje(valor: float) -> str:
    valor = max(0.0, min(1.0, float(valor)))
    return f"{valor * 100:.1f}%"


def numero(valor: Any, defecto: float = 0.0) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return defecto


def entero(valor: Any, defecto: int = 0) -> int:
    try:
        return int(valor)
    except (TypeError, ValueError):
        return defecto


def lista_segura(valor: Any) -> List[Any]:
    if isinstance(valor, list):
        return valor
    if valor is None:
        return []
    return [valor]


def diccionario_seguro(valor: Any) -> Dict[str, Any]:
    return valor if isinstance(valor, dict) else {}


# ================================================================
# MEMORIA Y ESTADO
# ================================================================


def incrementar_interacciones(memoria: Dict[str, Any]) -> Dict[str, Any]:
    memoria["interacciones"] = entero(memoria.get("interacciones", 0)) + 1
    return memoria


def obtener_gustos(memoria: Dict[str, Any]) -> List[str]:
    gustos = memoria.get("gustos", [])
    return gustos if isinstance(gustos, list) else []


def recordar_gusto(gusto: str) -> str:
    agregar_gusto(gusto)
    return f"He registrado: {gusto}"


def cargar_estado() -> Dict[str, Any]:
    if not RUTA_ESTADO.exists():
        return {}
    try:
        return json.loads(RUTA_ESTADO.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def guardar_estado(datos: Dict[str, Any]) -> bool:
    try:
        RUTA_ESTADO.parent.mkdir(parents=True, exist_ok=True)
        RUTA_ESTADO.write_text(json.dumps(datos, ensure_ascii=False, indent=4), encoding="utf-8")
        return True
    except OSError:
        return False


def sincronizar_estado_con_memoria(memoria: Dict[str, Any]) -> None:
    memoria.setdefault("tamy", {})["ultima_intencion"] = estado_tamy.get("ultima_intencion", "")
    memoria.setdefault("tamy", {})["ultimo_mensaje"] = estado_tamy.get("ultimo_mensaje", "")


def establecer_ultimo_mensaje(mensaje: str) -> None:
    estado_tamy["ultimo_mensaje"] = limitar_texto(mensaje)


def establecer_ultima_intencion(intencion: str) -> None:
    estado_tamy["ultima_intencion"] = intencion


def limpiar_estado_confirmacion() -> None:
    estado_tamy["candidatas"] = []
    estado_tamy["mensaje_original"] = ""
    estado_tamy["esperando_confirmacion"] = False


def guardar_turno(usuario: str, tamy: str, intencion: str = "", confirmada: Optional[bool] = None) -> None:
    estado_tamy.setdefault("turnos", []).append({
        "usuario": limitar_texto(usuario),
        "tamy": limitar_texto(tamy),
        "intencion": intencion,
        "confirmada": confirmada,
        "timestamp": ahora_texto(),
    })
    estado_tamy["turnos"] = estado_tamy["turnos"][-50:]


# ================================================================
# HIPÓTESIS
# ================================================================


def convertir_hipotesis(item: Dict[str, Any], posicion: int) -> HipotesisActiva:
    return HipotesisActiva(
        intencion=str(item.get("intencion", "")),
        puntuacion=numero(item.get("puntuacion", 0)),
        pregunta=str(item.get("pregunta", "")),
        confianza=numero(item.get("confianza", 0)),
        evidencias=lista_segura(item.get("evidencias", [])),
        posicion=posicion,
    )


def convertir_resultados(resultados: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(x) for x in resultados if isinstance(x, dict)]


def top_tres(resultados: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return convertir_resultados(list(resultados)[:MAX_HIPOTESIS])


def mejor_hipotesis(resultados: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return dict(resultados[0]) if resultados else None


def segunda_hipotesis(resultados: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return dict(resultados[1]) if len(resultados) > 1 else None


def tercera_hipotesis(resultados: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return dict(resultados[2]) if len(resultados) > 2 else None


def diferencia_hipotesis(resultados: Sequence[Dict[str, Any]]) -> float:
    if len(resultados) < 2:
        return float("inf")
    return numero(resultados[0].get("puntuacion", 0)) - numero(resultados[1].get("puntuacion", 0))


def hipotesis_ambiguas(resultados: Sequence[Dict[str, Any]]) -> bool:
    return diferencia_hipotesis(resultados) <= UMBRAL_CONFIRMACION


def requiere_confirmacion(intencion: str, resultados: Sequence[Dict[str, Any]]) -> bool:
    if intencion in INTENCIONES_CONFIRMACION:
        return True
    return hipotesis_ambiguas(resultados)


def construir_debug(resultados: Sequence[Dict[str, Any]]) -> List[str]:
    lineas: List[str] = []
    for i, item in enumerate(resultados[:MAX_HIPOTESIS], 1):
        lineas.append(
            f"{i}. {item.get('intencion')} -> {numero(item.get('puntuacion')):.3f} "
            f"| base={numero(item.get('puntuacion_base')):.3f} "
            f"| aprendizaje={numero(item.get('bono_aprendizaje')):.3f} "
            f"| confianza={numero(item.get('confianza')):.3f}"
        )
    return lineas


def imprimir_debug(resultados: Sequence[Dict[str, Any]]) -> None:
    print("\nDEBUG - HIPÓTESIS:")
    lineas = construir_debug(resultados)
    if not lineas:
        print("  Ninguna hipótesis encontrada.")
        return
    for linea in lineas:
        print(f"  {linea}")


# ================================================================
# RESPUESTAS FIJAS Y ACCIONES
# ================================================================


def respuesta_saludar(memoria: Dict[str, Any]) -> str:
    return "Hola. Soy TAMY. ¿Qué hacemos?"


def respuesta_identidad(memoria: Dict[str, Any]) -> str:
    try:
        return obtener_presentacion()
    except Exception:
        return f"Soy TAMY, versión {VERSION}."


def respuesta_hora(memoria: Dict[str, Any]) -> str:
    return f"La hora actual es {ahora().strftime('%H:%M:%S')}."


def respuesta_fecha(memoria: Dict[str, Any]) -> str:
    return f"Hoy es {ahora().strftime('%d/%m/%Y')}."


def respuesta_mes(memoria: Dict[str, Any]) -> str:
    meses = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
    ]
    return f"Estamos en {meses[ahora().month - 1]}."


def respuesta_año(memoria: Dict[str, Any]) -> str:
    return f"Estamos en el año {ahora().year}."


def respuesta_memoria(memoria: Dict[str, Any]) -> str:
    return (
        f"Tengo registradas {entero(memoria.get('interacciones', 0))} interacciones "
        f"y {len(obtener_gustos(memoria))} preferencias."
    )


def respuesta_gustos(memoria: Dict[str, Any]) -> str:
    gustos = obtener_gustos(memoria)
    if not gustos:
        return "Todavía no tengo preferencias registradas."
    return "Estas son mis preferencias registradas:\n" + "\n".join(f"• {g}" for g in gustos)


def respuesta_aprendizaje(memoria: Dict[str, Any]) -> str:
    return (
        "Entendido. Estoy aprendiendo a interpretar tus intenciones "
        "y a utilizar tus respuestas para mejorar mis futuras hipótesis."
    )


def respuesta_capacidades(memoria: Dict[str, Any]) -> str:
    return (
        "Actualmente puedo:\n"
        "• Conversar contigo\n"
        "• Detectar intenciones\n"
        "• Comparar hipótesis\n"
        "• Aprender de tus respuestas\n"
        "• Consultar memoria\n"
        "• Trabajar con fechas y horas\n"
        "• Prepararme para ejecutar acciones en el sistema"
    )


def respuesta_ayuda(memoria: Dict[str, Any]) -> str:
    return "Puedo ayudarte a conversar, consultar información, organizar ideas y desarrollar nuevas funciones."


def respuesta_trabajar(memoria: Dict[str, Any]) -> str:
    return "Perfecto. Puedo ayudarte a organizar lo que vas a trabajar."


def respuesta_proyecto(memoria: Dict[str, Any]) -> str:
    return "Puedo ayudarte a dividir un proyecto en objetivos, tareas y próximos pasos."


def respuesta_buscar(memoria: Dict[str, Any]) -> str:
    return "He identificado una intención de búsqueda. El módulo de búsqueda todavía no está conectado."


def respuesta_abrir_aplicacion(memoria: Dict[str, Any]) -> str:
    return "He identificado que quieres abrir una aplicación. El módulo de acciones todavía no está conectado."


def respuesta_abrir_archivo(memoria: Dict[str, Any]) -> str:
    return "He identificado que quieres trabajar con un archivo. El módulo de archivos todavía no está conectado."


def respuesta_crear_archivo(memoria: Dict[str, Any]) -> str:
    return "He identificado que quieres crear un archivo. El módulo de archivos todavía no está conectado."


def respuesta_guardar_memoria(memoria: Dict[str, Any]) -> str:
    return "Puedo guardarlo en memoria. Dime exactamente qué dato quieres que recuerde."


def respuesta_tareas(memoria: Dict[str, Any]) -> str:
    return "Todavía no tengo conectado un gestor de tareas."


def respuesta_curiosidad(memoria: Dict[str, Any]) -> str:
    return "Adelante. Hazme la pregunta."


def respuesta_despedida(memoria: Dict[str, Any]) -> str:
    return "Entendido. Estaré aquí cuando regreses."


def respuesta_agradecimiento(memoria: Dict[str, Any]) -> str:
    return "De nada. Seguimos trabajando."


HANDLERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "saludar": respuesta_saludar,
    "identidad": respuesta_identidad,
    "hora": respuesta_hora,
    "fecha": respuesta_fecha,
    "mes": respuesta_mes,
    "año": respuesta_año,
    "memoria": respuesta_memoria,
    "gustos": respuesta_gustos,
    "aprendizaje": respuesta_aprendizaje,
    "capacidades": respuesta_capacidades,
    "ayuda": respuesta_ayuda,
    "trabajar": respuesta_trabajar,
    "proyecto": respuesta_proyecto,
    "buscar": respuesta_buscar,
    "abrir_aplicacion": respuesta_abrir_aplicacion,
    "abrir_archivo": respuesta_abrir_archivo,
    "crear_archivo": respuesta_crear_archivo,
    "guardar_memoria": respuesta_guardar_memoria,
    "tareas": respuesta_tareas,
    "curiosidad": respuesta_curiosidad,
    "despedida": respuesta_despedida,
    "agradecimiento": respuesta_agradecimiento,
}


def ejecutar_intencion(intencion: str, memoria: Dict[str, Any]) -> Optional[str]:
    handler = HANDLERS.get(intencion)
    if handler is None:
        return None
    try:
        return handler(memoria)
    except Exception as exc:
        return f"Detecté la intención {formatear_intencion(intencion)}, pero ocurrió un problema interno: {exc}"


# ================================================================
# APRENDIZAJE
# ================================================================


def aplicar_aprendizaje(mensaje: str, resultados: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        return motor_aprendizaje.ajustar_resultados(mensaje, resultados)
    except Exception as exc:
        print(f"DEBUG - aprendizaje no aplicado: {exc}")
        return convertir_resultados(resultados)


def aprender_confirmacion(mensaje: str, resultados: Sequence[Dict[str, Any]]) -> List[Any]:
    if not resultados:
        return []
    try:
        intencion = resultados[0].get("intencion")
        if not intencion:
            return []
        return motor_aprendizaje.aprender_de_resultados(
            mensaje=mensaje,
            resultados=resultados,
            intencion_correcta=intencion,
        )
    except Exception as exc:
        print(f"DEBUG - aprendizaje positivo falló: {exc}")
        return []


def penalizar_hipotesis(mensaje: str, resultado: Dict[str, Any]) -> Any:
    intencion = resultado.get("intencion")
    if not intencion:
        return None
    try:
        return motor_aprendizaje.penalizar(
            intencion=intencion,
            mensaje=mensaje,
            motivo="primera_hipotesis_rechazada",
            confianza_previa=numero(resultado.get("confianza", 0)),
        )
    except TypeError:
        try:
            return motor_aprendizaje.penalizar(
                intencion=intencion,
                mensaje=mensaje,
            )
        except Exception:
            return None
    except Exception:
        return None


def mostrar_aprendizaje(eventos: Sequence[Any]) -> None:
    if not eventos:
        return
    print("\n🧠 APRENDIZAJE:")
    for evento in eventos:
        intencion = getattr(evento, "intencion", "?")
        anterior = getattr(evento, "peso_anterior", "?")
        nuevo = getattr(evento, "peso_nuevo", "?")
        recompensa = getattr(evento, "recompensa", "?")
        print(f"  {intencion}: {anterior} -> {nuevo} | recompensa={recompensa}")


# ================================================================
# CONFIRMACIÓN
# ================================================================


def preparar_confirmacion(mensaje: str, resultados: Sequence[Dict[str, Any]]) -> str:
    estado_tamy["candidatas"] = top_tres(resultados)
    estado_tamy["mensaje_original"] = mensaje
    estado_tamy["esperando_confirmacion"] = bool(estado_tamy["candidatas"])
    return elegir_pregunta(estado_tamy["candidatas"]) or "¿Es correcta esta interpretación?"


def confirmar_primera_hipotesis(memoria: Dict[str, Any]) -> Optional[str]:
    candidatas = estado_tamy.get("candidatas", [])
    if not candidatas:
        return None
    intencion = candidatas[0].get("intencion", "")
    eventos = aprender_confirmacion(estado_tamy.get("mensaje_original", ""), candidatas)
    mostrar_aprendizaje(eventos)
    limpiar_estado_confirmacion()
    establecer_ultima_intencion(intencion)
    return ejecutar_intencion(intencion, memoria)


def rechazar_primera_hipotesis() -> Optional[str]:
    candidatas = estado_tamy.get("candidatas", [])
    if not candidatas:
        limpiar_estado_confirmacion()
        return None
    rechazada = candidatas.pop(0)
    penalizar_hipotesis(estado_tamy.get("mensaje_original", ""), rechazada)
    if not candidatas:
        limpiar_estado_confirmacion()
        return None
    estado_tamy["candidatas"] = candidatas
    estado_tamy["esperando_confirmacion"] = True
    return elegir_pregunta(candidatas)


def procesar_confirmacion(mensaje: str, memoria: Dict[str, Any]) -> Optional[str]:
    if not estado_tamy.get("esperando_confirmacion"):
        return None
    if es_si(mensaje):
        return confirmar_primera_hipotesis(memoria)
    if es_no(mensaje):
        siguiente = rechazar_primera_hipotesis()
        if siguiente:
            return siguiente
        return "Entendido. Ninguna de mis hipótesis coincidió. Explícame un poco más."
    return "Necesito saber si esa interpretación es correcta. Puedes responder 'sí' o 'no'."


# ================================================================
# COMANDOS ESPECIALES DEL CEREBRO
# ================================================================


def comando_estado() -> str:
    return (
        f"TAMY {VERSION}\n"
        f"Modo: {estado_tamy.get('modo')}\n"
        f"Interacción en espera: {estado_tamy.get('esperando_confirmacion')}\n"
        f"Hipótesis activas: {len(estado_tamy.get('candidatas', []))}"
    )


def comando_hipotesis() -> str:
    candidatas = estado_tamy.get("candidatas", [])
    if not candidatas:
        return "No tengo hipótesis activas en este momento."
    return "\n".join(
        f"{i}. {x.get('intencion')} -> {numero(x.get('puntuacion')):.3f}"
        for i, x in enumerate(candidatas, 1)
    )


def comando_diagnostico() -> str:
    try:
        datos = motor_aprendizaje.diagnostico()
    except Exception as exc:
        return f"No pude obtener el diagnóstico: {exc}"
    estadisticas = datos.get("estadisticas", {})
    return (
        "Diagnóstico de aprendizaje:\n"
        f"Eventos: {estadisticas.get('eventos', 0)}\n"
        f"Ejemplos: {estadisticas.get('ejemplos', 0)}\n"
        f"Intenciones aprendidas: {estadisticas.get('intenciones', 0)}"
    )


def comando_ranking() -> str:
    try:
        ranking = motor_aprendizaje.ranking_intenciones(5)
    except Exception as exc:
        return f"No pude obtener el ranking: {exc}"
    if not ranking:
        return "Todavía no tengo un ranking de aprendizaje."
    return "\n".join(
        f"{i}. {x.get('intencion')} -> {numero(x.get('peso')):.3f}"
        for i, x in enumerate(ranking, 1)
    )


def comando_memoria_aprendizaje() -> str:
    try:
        resumen = motor_aprendizaje.resumen_texto()
        return resumen
    except Exception as exc:
        return f"No pude consultar la memoria de aprendizaje: {exc}"


def procesar_comando_especial(mensaje: str) -> Optional[str]:
    valor = normalizar(mensaje)
    if valor == "estado tamy":
        return comando_estado()
    if valor == "ver hipotesis":
        return comando_hipotesis()
    if valor == "diagnostico":
        return comando_diagnostico()
    if valor == "ver ranking":
        return comando_ranking()
    if valor == "memoria de aprendizaje":
        return comando_memoria_aprendizaje()
    return None


# ================================================================
# MEMORIA DIRECTA
# ================================================================


def procesar_memoria_directa(mensaje: str, memoria: Dict[str, Any]) -> Optional[str]:
    valor = normalizar(mensaje)
    if valor in {"que recuerdas", "qué recuerdas"}:
        return respuesta_memoria(memoria)
    if valor in {"que te gusta", "qué te gusta"}:
        return respuesta_gustos(memoria)
    if valor.startswith("te gusta "):
        gusto = valor.replace("te gusta ", "", 1).strip()
        if gusto:
            return recordar_gusto(gusto)
    return None


# ================================================================
# PROCESAMIENTO PRINCIPAL
# ================================================================


def registrar_entrada(mensaje: str, memoria: Dict[str, Any]) -> None:
    establecer_ultimo_mensaje(mensaje)
    memoria.setdefault("tamy", {})["ultimo_mensaje"] = limitar_texto(mensaje)
    motor_aprendizaje.memoria.setdefault("sesion", {})["ultimo_mensaje"] = normalizar(mensaje)


def actualizar_memoria_interaccion(memoria: Dict[str, Any]) -> None:
    incrementar_interacciones(memoria)
    sincronizar_estado_con_memoria(memoria)
    guardar_memoria(memoria)


def registrar_respuesta(mensaje: str, respuesta: str, intencion: str = "", confirmada: Optional[bool] = None) -> None:
    guardar_turno(mensaje, respuesta, intencion, confirmada)
    estado_tamy["ultimo_resultado"] = respuesta


def procesar_texto(mensaje_usuario: str) -> str:
    memoria = cargar_memoria()
    actualizar_memoria_interaccion(memoria)
    mensaje = normalizar(mensaje_usuario)
    if es_vacio(mensaje):
        respuesta = "No recibí ningún mensaje."
        registrar_respuesta(mensaje_usuario, respuesta)
        return respuesta

    registrar_entrada(mensaje_usuario, memoria)

    especial = procesar_comando_especial(mensaje_usuario)
    if especial is not None:
        registrar_respuesta(mensaje_usuario, especial)
        return especial

    confirmacion = procesar_confirmacion(mensaje_usuario, memoria)
    if confirmacion is not None:
        registrar_respuesta(mensaje_usuario, confirmacion, estado_tamy.get("ultima_intencion", ""))
        return confirmacion

    directa = procesar_memoria_directa(mensaje_usuario, memoria)
    if directa is not None:
        registrar_respuesta(mensaje_usuario, directa)
        return directa

    try:
        resultados_base = analizar(mensaje_usuario)
    except Exception as exc:
        respuesta = f"Ocurrió un error al analizar tu mensaje: {exc}"
        registrar_respuesta(mensaje_usuario, respuesta)
        return respuesta

    resultados = aplicar_aprendizaje(mensaje_usuario, resultados_base)
    resultados = top_tres(resultados)
    imprimir_debug(resultados)

    if not resultados:
        respuesta = (
            "Todavía no tengo una hipótesis clara sobre lo que quieres decir. "
            "Puedes explicármelo de otra manera."
        )
        registrar_respuesta(mensaje_usuario, respuesta)
        return respuesta

    mejor = mejor_hipotesis(resultados) or {}
    intencion = str(mejor.get("intencion", ""))
    puntuacion = numero(mejor.get("puntuacion", 0))

    if puntuacion >= UMBRAL_RESPUESTA_DIRECTA and not requiere_confirmacion(intencion, resultados):
        respuesta = ejecutar_intencion(intencion, memoria)
        if respuesta:
            establecer_ultima_intencion(intencion)
            limpiar_estado_confirmacion()
            registrar_respuesta(mensaje_usuario, respuesta, intencion, True)
            return respuesta

    pregunta = preparar_confirmacion(mensaje_usuario, resultados)
    respuesta = f"Tengo una posible interpretación: {intencion}.\n{pregunta}"
    registrar_respuesta(mensaje_usuario, respuesta, intencion, None)
    return respuesta


# ================================================================
# IMÁGENES
# ================================================================


def analizar_imagen_y_aprender(ruta_archivo: str, comentario: str = "") -> str:
    nombre = os.path.basename(ruta_archivo)
    if not os.path.exists(ruta_archivo):
        return f"No encontré la imagen: {nombre}."
    if comentario:
        comentario = limitar_texto(comentario, 1000)
        return (
            f"He recibido la imagen '{nombre}'. "
            f"También registraré este contexto: '{comentario}'. "
            "El módulo avanzado de visión todavía está en desarrollo."
        )
    return (
        f"He recibido la imagen '{nombre}'. "
        "El módulo avanzado de visión todavía está en desarrollo."
    )


# ================================================================
# API DE ENTRENAMIENTO MANUAL
# ================================================================


def enseñar(mensaje: str, intencion: str, correcto: bool = True) -> str:
    try:
        resultado = motor_aprendizaje.entrenar(mensaje, intencion, correcto=correcto)
        return (
            f"Aprendizaje registrado: {intencion} | "
            f"peso {getattr(resultado, 'peso_anterior', 0):.3f} -> "
            f"{getattr(resultado, 'peso_nuevo', 0):.3f}"
        )
    except Exception as exc:
        return f"No pude registrar el entrenamiento: {exc}"


def enseñar_par(mensaje: str, correcta: str, incorrecta: str = "") -> str:
    try:
        resultados = motor_aprendizaje.entrenar_par(mensaje, correcta, incorrecta or None)
        return "\n".join(
            f"{getattr(r, 'intencion', '?')}: {getattr(r, 'peso_anterior', 0):.3f} -> {getattr(r, 'peso_nuevo', 0):.3f}"
            for r in resultados
        )
    except Exception as exc:
        return f"No pude entrenar el par: {exc}"


def enseñar_palabra(palabra: str, intencion: str, valor: float = 2.0) -> str:
    try:
        nuevo = motor_aprendizaje.enseñar_palabra(palabra, intencion, valor)
        return f"Palabra aprendida: {palabra} -> {intencion} = {nuevo:.3f}"
    except Exception as exc:
        return f"No pude enseñar la palabra: {exc}"


def enseñar_frase(frase: str, intencion: str, valor: float = 3.0) -> str:
    try:
        nuevo = motor_aprendizaje.enseñar_frase(frase, intencion, valor)
        return f"Frase aprendida: {intencion} = {nuevo:.3f}"
    except Exception as exc:
        return f"No pude enseñar la frase: {exc}"


def enseñar_correccion(original: str, corregida: str, confianza: float = 1.0) -> str:
    try:
        motor_aprendizaje.enseñar_correccion(original, corregida, confianza)
        return f"Corrección aprendida: {original} -> {corregida}"
    except Exception as exc:
        return f"No pude guardar la corrección: {exc}"


# ================================================================
# DIAGNÓSTICO Y CONSULTA
# ================================================================


def diagnostico() -> Dict[str, Any]:
    try:
        return motor_aprendizaje.diagnostico()
    except Exception as exc:
        return {"error": str(exc)}


def ranking_aprendizaje(limite: int = 10) -> List[Dict[str, Any]]:
    try:
        return motor_aprendizaje.ranking_intenciones(limite)
    except Exception:
        return []


def conocimiento_adquirido(minimo: float = 1.0) -> List[Dict[str, Any]]:
    try:
        return motor_aprendizaje.conocimiento_adquirido(minimo)
    except Exception:
        return []


def resumen_cerebro() -> Dict[str, Any]:
    return {
        "nombre": NOMBRE,
        "version": VERSION,
        "modo": estado_tamy.get("modo"),
        "esperando_confirmacion": estado_tamy.get("esperando_confirmacion"),
        "hipotesis": top_tres(estado_tamy.get("candidatas", [])),
        "ultima_intencion": estado_tamy.get("ultima_intencion", ""),
        "ultimo_mensaje": estado_tamy.get("ultimo_mensaje", ""),
    }


# ================================================================
# FUNCIONES DE COMPATIBILIDAD
# ================================================================


def _es_afirmacion(mensaje: str) -> bool:
    return es_si(mensaje)


def _es_negacion(mensaje: str) -> bool:
    return es_no(mensaje)


def ejecutar_accion(intencion: str, memoria: Optional[Dict[str, Any]] = None) -> Optional[str]:
    return ejecutar_intencion(intencion, memoria or cargar_memoria())


def procesar_mensaje(mensaje: str) -> str:
    return procesar_texto(mensaje)


# ================================================================
# BIBLIOTECA DE RESPUESTAS AUXILIARES
# ================================================================


def herramienta_auxiliar_001(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 001 del subsistema memoria.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_001",
        "subsistema": "memoria",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_002(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 002 del subsistema hipotesis.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_002",
        "subsistema": "hipotesis",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_003(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 003 del subsistema aprendizaje.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_003",
        "subsistema": "aprendizaje",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_004(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 004 del subsistema respuesta.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_004",
        "subsistema": "respuesta",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_005(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 005 del subsistema diagnostico.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_005",
        "subsistema": "diagnostico",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_006(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 006 del subsistema contexto.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_006",
        "subsistema": "contexto",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_007(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 007 del subsistema memoria.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_007",
        "subsistema": "memoria",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_008(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 008 del subsistema hipotesis.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_008",
        "subsistema": "hipotesis",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_009(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 009 del subsistema aprendizaje.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_009",
        "subsistema": "aprendizaje",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_010(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 010 del subsistema respuesta.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_010",
        "subsistema": "respuesta",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_011(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 011 del subsistema diagnostico.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_011",
        "subsistema": "diagnostico",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_012(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 012 del subsistema contexto.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_012",
        "subsistema": "contexto",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_013(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 013 del subsistema memoria.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_013",
        "subsistema": "memoria",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_014(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 014 del subsistema hipotesis.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_014",
        "subsistema": "hipotesis",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_015(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 015 del subsistema aprendizaje.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_015",
        "subsistema": "aprendizaje",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_016(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 016 del subsistema respuesta.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_016",
        "subsistema": "respuesta",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_017(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 017 del subsistema diagnostico.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_017",
        "subsistema": "diagnostico",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_018(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 018 del subsistema contexto.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_018",
        "subsistema": "contexto",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_019(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 019 del subsistema memoria.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_019",
        "subsistema": "memoria",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_020(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 020 del subsistema hipotesis.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_020",
        "subsistema": "hipotesis",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_021(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 021 del subsistema aprendizaje.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_021",
        "subsistema": "aprendizaje",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_022(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 022 del subsistema respuesta.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_022",
        "subsistema": "respuesta",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_023(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 023 del subsistema diagnostico.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_023",
        "subsistema": "diagnostico",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_024(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 024 del subsistema contexto.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_024",
        "subsistema": "contexto",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_025(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 025 del subsistema memoria.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_025",
        "subsistema": "memoria",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_026(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 026 del subsistema hipotesis.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_026",
        "subsistema": "hipotesis",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_027(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 027 del subsistema aprendizaje.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_027",
        "subsistema": "aprendizaje",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_028(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 028 del subsistema respuesta.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_028",
        "subsistema": "respuesta",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_029(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 029 del subsistema diagnostico.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_029",
        "subsistema": "diagnostico",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_030(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 030 del subsistema contexto.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_030",
        "subsistema": "contexto",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_031(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 031 del subsistema memoria.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_031",
        "subsistema": "memoria",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_032(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 032 del subsistema hipotesis.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_032",
        "subsistema": "hipotesis",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_033(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 033 del subsistema aprendizaje.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_033",
        "subsistema": "aprendizaje",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_034(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 034 del subsistema respuesta.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_034",
        "subsistema": "respuesta",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_035(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 035 del subsistema diagnostico.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_035",
        "subsistema": "diagnostico",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_036(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 036 del subsistema contexto.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_036",
        "subsistema": "contexto",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_037(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 037 del subsistema memoria.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_037",
        "subsistema": "memoria",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_038(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 038 del subsistema hipotesis.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_038",
        "subsistema": "hipotesis",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_039(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 039 del subsistema aprendizaje.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_039",
        "subsistema": "aprendizaje",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_040(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 040 del subsistema respuesta.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_040",
        "subsistema": "respuesta",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_041(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 041 del subsistema diagnostico.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_041",
        "subsistema": "diagnostico",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_042(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 042 del subsistema contexto.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_042",
        "subsistema": "contexto",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


def herramienta_auxiliar_043(valor: Any = None) -> Dict[str, Any]:
    """Herramienta auxiliar 043 del subsistema memoria.

    Estas funciones forman una capa estable de utilidades para futuras
    ampliaciones del cerebro sin obligar a tocar procesar_texto().
    """
    resultado = {
        "herramienta": "herramienta_auxiliar_043",
        "subsistema": "memoria",
        "entrada": valor,
        "timestamp": ahora_texto(),
    }
    if isinstance(valor, str):
        resultado["normalizado"] = normalizar(valor)
        resultado["tokens"] = tokens(valor)
    elif isinstance(valor, dict):
        resultado["claves"] = list(valor.keys())[:20]
    else:
        resultado["tipo"] = type(valor).__name__
    return resultado


# ================================================================
# ROUTER EXPERIMENTAL DE TAMY
# ================================================================


ROUTERS: Dict[str, Callable[[str], str]] = {
    "estado": lambda m: comando_estado(),
    "hipotesis": lambda m: comando_hipotesis(),
    "diagnostico": lambda m: comando_diagnostico(),
    "ranking": lambda m: comando_ranking(),
}


def ejecutar_router(nombre: str, argumento: str = "") -> str:
    handler = ROUTERS.get(normalizar(nombre))
    if handler is None:
        return "No existe ese comando del router."
    return handler(argumento)


def guardar_snapshot() -> bool:
    datos = resumen_cerebro()
    return guardar_estado(datos)


def restaurar_snapshot() -> Dict[str, Any]:
    return cargar_estado()


def resetear_conversacion() -> None:
    estado_tamy["candidatas"] = []
    estado_tamy["mensaje_original"] = ""
    estado_tamy["esperando_confirmacion"] = False
    estado_tamy["ultimo_resultado"] = None


def cambiar_modo(modo: str) -> str:
    modo = normalizar(modo) or "normal"
    estado_tamy["modo"] = modo
    return f"Modo TAMY cambiado a: {modo}"


def obtener_modo() -> str:
    return str(estado_tamy.get("modo", "normal"))


def obtener_ultima_intencion() -> str:
    return str(estado_tamy.get("ultima_intencion", ""))


def obtener_historial_turnos() -> List[Dict[str, Any]]:
    return list(estado_tamy.get("turnos", []))


def limpiar_historial_turnos() -> None:
    estado_tamy["turnos"] = []


def exportar_estado(ruta: str = "data/tamy_brain_export.json") -> bool:
    datos = {
        "estado": estado_tamy,
        "aprendizaje": diagnostico(),
        "timestamp": ahora_texto(),
    }
    try:
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(datos, ensure_ascii=False, indent=4), encoding="utf-8")
        return True
    except OSError:
        return False


def seguridad_basica(mensaje: str) -> Dict[str, Any]:
    valor = normalizar(mensaje)
    palabras_peligrosas = {"borrar", "eliminar", "formatear", "apagar", "destruir"}
    detectadas = [x for x in tokens(valor) if x in palabras_peligrosas]
    return {
        "requiere_confirmacion": bool(detectadas),
        "detectadas": detectadas,
        "motivo": "acción sensible" if detectadas else "ninguno",
    }


def preparar_accion(intencion: str, argumento: str = "") -> Dict[str, Any]:
    seguridad = seguridad_basica(argumento)
    return {
        "intencion": intencion,
        "argumento": argumento,
        "seguridad": seguridad,
        "puede_ejecutar": not seguridad["requiere_confirmacion"],
    }


def planificar(intenciones: Sequence[str]) -> List[Dict[str, Any]]:
    plan = []
    for posicion, intencion in enumerate(intenciones, 1):
        plan.append({
            "paso": posicion,
            "intencion": intencion,
            "estado": "pendiente",
        })
    return plan


def marcar_paso(plan: List[Dict[str, Any]], indice: int, estado: str) -> List[Dict[str, Any]]:
    copia = [dict(x) for x in plan]
    if 0 <= indice < len(copia):
        copia[indice]["estado"] = estado
    return copia


# ================================================================
# ARRANQUE
# ================================================================


def inicializar() -> Dict[str, Any]:
    datos = cargar_estado()
    if isinstance(datos, dict):
        estado_tamy.update({
            "modo": datos.get("modo", estado_tamy.get("modo", "normal")),
            "ultima_intencion": datos.get("ultima_intencion", ""),
            "ultimo_mensaje": datos.get("ultimo_mensaje", ""),
        })
    return resumen_cerebro()


INICIALIZADO = inicializar()