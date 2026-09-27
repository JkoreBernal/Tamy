"""
TAMY - MOTOR DE APRENDIZAJE v1.0
================================

Segundo motor cognitivo de TAMY.

El motor de hipótesis intenta responder:
    "¿Qué intención podría tener este mensaje?"

Este motor intenta responder:
    "¿Qué aprendí de la respuesta del usuario?"

Principio general:

    mensaje
       ↓
    hipótesis
       ↓
    respuesta del usuario
       ↓
    recompensa / penalización
       ↓
    memoria de aprendizaje
       ↓
    nuevos pesos
       ↓
    futuras hipótesis mejor ordenadas

NO es todavía una red neuronal. Es un motor de aprendizaje por refuerzo
heurístico y persistente. La ventaja para el proyecto de TAMY es que sus
reglas son transparentes: podemos inspeccionar por qué aumentó o bajó una
preferencia.

API principal:
    MotorAprendizaje
    AprendizajeResultado
    crear_motor

Funciones de conveniencia:
    recompensar
    penalizar
    registrar_feedback
    ajustar_resultados
    aprender_de_mensaje

Archivo persistente por defecto:
    data/tamy_learning.json

El módulo no ejecuta acciones del sistema operativo y no modifica por sí
mismo motor_hipotesis.py. Entrega ajustes que tamy_brain.py puede aplicar.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .motor_hipotesis import normalizar, tokens
except ImportError:
    from motor_hipotesis import normalizar, tokens


# ============================================================================
# CONSTANTES GENERALES
# ============================================================================

VERSION = "1.0"
SCHEMA_VERSION = 1
DEFAULT_MEMORY_PATH = Path("data/tamy_learning.json")
DEFAULT_LEARNING_RATE = 0.50
DEFAULT_REWARD = 2.0
DEFAULT_PENALTY = -2.0
DEFAULT_STRONG_REWARD = 3.0
DEFAULT_STRONG_PENALTY = -3.0
DEFAULT_DECAY = 0.995
DEFAULT_MAX_WEIGHT = 12.0
DEFAULT_MIN_WEIGHT = -12.0
DEFAULT_MAX_HISTORY = 1000
DEFAULT_MAX_EXAMPLES = 500
DEFAULT_MAX_ASSOCIATIONS = 1000
DEFAULT_MAX_CORRECTIONS = 1000
DEFAULT_MIN_CONFIDENCE = 0.20
DEFAULT_CONFIRMATION_GAP = 1.50
DEFAULT_EXPLORATION = 0.10


# ============================================================================
# TIPOS DE DATOS
# ============================================================================

@dataclass
class AprendizajeResultado:
    """Resultado de una operación de aprendizaje."""

    mensaje: str
    intencion: str
    recompensa: float
    peso_anterior: float
    peso_nuevo: float
    motivo: str
    timestamp: str
    evento_id: str
    confianza_previa: float = 0.0
    confianza_nueva: float = 0.0
    evidencias: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Feedback:
    """Representa la respuesta del usuario a una hipótesis."""

    tipo: str
    mensaje: str
    intencion: str
    valor: float
    timestamp: str
    evento_id: str
    contexto: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EjemploAprendido:
    """Ejemplo de lenguaje asociado con una intención."""

    mensaje: str
    intencion: str
    recompensa_total: float = 0.0
    apariciones: int = 0
    aciertos: int = 0
    rechazos: int = 0
    ultima_actualizacion: str = ""

    @property
    def tasa_acierto(self) -> float:
        if self.apariciones <= 0:
            return 0.0
        return self.aciertos / self.apariciones

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tasa_acierto"] = round(self.tasa_acierto, 4)
        return data


@dataclass
class AsociacionContextual:
    """Asociación entre una intención y otra intención previa."""

    anterior: str
    siguiente: str
    peso: float = 0.0
    apariciones: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# UTILIDADES DE TEXTO Y TIEMPO
# ============================================================================


def ahora_iso() -> str:
    """Devuelve la hora actual en UTC con formato ISO."""
    return datetime.now(timezone.utc).isoformat()


def generar_id() -> str:
    """Genera un identificador corto para un evento."""
    return uuid.uuid4().hex[:12]


def limitar(valor: float, minimo: float, maximo: float) -> float:
    """Limita un número a un rango."""
    return max(minimo, min(maximo, valor))


def sigmoide(valor: float) -> float:
    """Transforma un valor arbitrario a un rango aproximado 0..1."""
    try:
        return 1.0 / (1.0 + math.exp(-valor))
    except OverflowError:
        return 0.0 if valor < 0 else 1.0


def logaritmo_seguro(valor: float) -> float:
    """Logaritmo protegido para cálculos heurísticos."""
    return math.log(max(valor, 1e-9))


def normalizar_intencion(nombre: str) -> str:
    """Normaliza el identificador de una intención."""
    return normalizar(nombre).replace(" ", "_")


def normalizar_mensaje(mensaje: str) -> str:
    """Normaliza un mensaje antes de guardarlo como ejemplo."""
    return " ".join(normalizar(mensaje).split())


def extraer_tokens_utiles(mensaje: str) -> List[str]:
    """Obtiene tokens no vacíos y suficientemente informativos."""
    resultado: List[str] = []
    for palabra in tokens(mensaje):
        if not palabra:
            continue
        if len(palabra) <= 1:
            continue
        resultado.append(palabra)
    return resultado


def clave_token(token: str) -> str:
    """Crea una clave estable para una palabra aprendida."""
    return normalizar(token)


def clave_frase(mensaje: str) -> str:
    """Crea una clave estable para una frase completa."""
    return normalizar_mensaje(mensaje)


# ============================================================================
# ESTADO INICIAL
# ============================================================================


def estado_vacio() -> Dict[str, Any]:
    """Construye una memoria de aprendizaje nueva."""
    return {
        "schema_version": SCHEMA_VERSION,
        "version_motor": VERSION,
        "creado": ahora_iso(),
        "actualizado": ahora_iso(),
        "total_eventos": 0,
        "total_recompensas": 0.0,
        "total_penalizaciones": 0.0,
        "racha_aciertos": 0,
        "mejor_racha": 0,
        "racha_errores": 0,
        "intenciones": {},
        "palabras": {},
        "frases": {},
        "ejemplos": {},
        "correcciones": {},
        "asociaciones": {},
        "historial": [],
        "sesion": {
            "ultima_intencion": None,
            "ultima_recompensa": 0.0,
            "ultimo_mensaje": "",
            "ultima_pregunta": "",
            "ultimo_evento": None,
        },
        "config": {
            "learning_rate": DEFAULT_LEARNING_RATE,
            "reward": DEFAULT_REWARD,
            "penalty": DEFAULT_PENALTY,
            "strong_reward": DEFAULT_STRONG_REWARD,
            "strong_penalty": DEFAULT_STRONG_PENALTY,
            "decay": DEFAULT_DECAY,
            "max_weight": DEFAULT_MAX_WEIGHT,
            "min_weight": DEFAULT_MIN_WEIGHT,
            "max_history": DEFAULT_MAX_HISTORY,
            "max_examples": DEFAULT_MAX_EXAMPLES,
            "max_associations": DEFAULT_MAX_ASSOCIATIONS,
            "max_corrections": DEFAULT_MAX_CORRECTIONS,
            "min_confidence": DEFAULT_MIN_CONFIDENCE,
            "confirmation_gap": DEFAULT_CONFIRMATION_GAP,
            "exploration": DEFAULT_EXPLORATION,
        },
    }


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================


class MotorAprendizaje:
    """Motor de aprendizaje por refuerzo heurístico para TAMY."""

    def __init__(
        self,
        ruta_memoria: os.PathLike[str] | str = DEFAULT_MEMORY_PATH,
        learning_rate: Optional[float] = None,
        max_history: Optional[int] = None,
    ) -> None:
        self.ruta_memoria = Path(ruta_memoria)
        self.memoria: Dict[str, Any] = estado_vacio()
        self.cargar()
        if learning_rate is not None:
            self.configurar(learning_rate=learning_rate)
        if max_history is not None:
            self.configurar(max_history=max_history)

    # ========================================================================
    # CARGA Y GUARDADO
    # ========================================================================

    def cargar(self) -> Dict[str, Any]:
        """Carga la memoria desde JSON si existe."""
        if not self.ruta_memoria.exists():
            self.memoria = estado_vacio()
            return self.memoria

        try:
            with self.ruta_memoria.open("r", encoding="utf-8") as archivo:
                datos = json.load(archivo)
        except (OSError, json.JSONDecodeError):
            self.memoria = estado_vacio()
            return self.memoria

        self.memoria = self._fusionar_con_estado_base(datos)
        return self.memoria

    def guardar(self) -> bool:
        """Guarda la memoria en JSON."""
        self.ruta_memoria.parent.mkdir(parents=True, exist_ok=True)
        self.memoria["actualizado"] = ahora_iso()

        temporal = self.ruta_memoria.with_suffix(
            self.ruta_memoria.suffix + ".tmp"
        )

        try:
            with temporal.open("w", encoding="utf-8") as archivo:
                json.dump(
                    self.memoria,
                    archivo,
                    indent=4,
                    ensure_ascii=False,
                )
            temporal.replace(self.ruta_memoria)
            return True
        except OSError:
            return False

    def recargar(self) -> Dict[str, Any]:
        """Recarga la memoria ignorando cambios en RAM."""
        return self.cargar()

    def _fusionar_con_estado_base(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Completa campos faltantes sin destruir datos antiguos."""
        base = estado_vacio()
        if not isinstance(datos, dict):
            return base

        for clave, valor in datos.items():
            base[clave] = valor

        for clave, valor in estado_vacio()["config"].items():
            base.setdefault("config", {}).setdefault(clave, valor)

        base.setdefault("intenciones", {})
        base.setdefault("palabras", {})
        base.setdefault("frases", {})
        base.setdefault("ejemplos", {})
        base.setdefault("correcciones", {})
        base.setdefault("asociaciones", {})
        base.setdefault("historial", [])
        base.setdefault("sesion", {})
        return base

    # ========================================================================
    # CONFIGURACIÓN
    # ========================================================================

    def configurar(self, **valores: Any) -> Dict[str, Any]:
        """Actualiza parámetros del motor de aprendizaje."""
        config = self.memoria.setdefault("config", {})
        claves_validas = {
            "learning_rate",
            "reward",
            "penalty",
            "strong_reward",
            "strong_penalty",
            "decay",
            "max_weight",
            "min_weight",
            "max_history",
            "max_examples",
            "max_associations",
            "max_corrections",
            "min_confidence",
            "confirmation_gap",
            "exploration",
        }

        for clave, valor in valores.items():
            if clave in claves_validas:
                config[clave] = valor

        self.guardar()
        return deepcopy(config)

    def config(self) -> Dict[str, Any]:
        """Devuelve una copia de la configuración."""
        return deepcopy(self.memoria.get("config", {}))

    def _cfg(self, clave: str, defecto: Any = None) -> Any:
        return self.memoria.get("config", {}).get(clave, defecto)

    # ========================================================================
    # PESOS DE INTENCIONES
    # ========================================================================

    def _obtener_intencion(self, intencion: str) -> Dict[str, Any]:
        nombre = normalizar_intencion(intencion)
        tabla = self.memoria.setdefault("intenciones", {})
        if nombre not in tabla:
            tabla[nombre] = {
                "peso": 0.0,
                "aciertos": 0,
                "errores": 0,
                "apariciones": 0,
                "recompensa_total": 0.0,
                "ultima_actualizacion": ahora_iso(),
            }
        return tabla[nombre]

    def peso_intencion(self, intencion: str) -> float:
        """Obtiene el peso aprendido de una intención."""
        datos = self._obtener_intencion(intencion)
        return float(datos.get("peso", 0.0))

    def ajustar_peso_intencion(
        self,
        intencion: str,
        delta: float,
        motivo: str = "ajuste_manual",
    ) -> float:
        """Ajusta el peso global de una intención."""
        datos = self._obtener_intencion(intencion)
        anterior = float(datos.get("peso", 0.0))
        nuevo = limitar(
            anterior + delta * float(self._cfg("learning_rate", DEFAULT_LEARNING_RATE)),
            float(self._cfg("min_weight", DEFAULT_MIN_WEIGHT)),
            float(self._cfg("max_weight", DEFAULT_MAX_WEIGHT)),
        )
        datos["peso"] = round(nuevo, 6)
        datos["ultima_actualizacion"] = ahora_iso()
        datos["ultimo_motivo"] = motivo
        return datos["peso"]

    def aplicar_decay(self) -> None:
        """Reduce lentamente pesos aprendidos excesivamente antiguos."""
        decay = float(self._cfg("decay", DEFAULT_DECAY))
        for datos in self.memoria.get("intenciones", {}).values():
            peso = float(datos.get("peso", 0.0))
            datos["peso"] = round(peso * decay, 6)
        for datos in self.memoria.get("palabras", {}).values():
            for intencion, peso in list(datos.items()):
                if intencion.startswith("_"):
                    continue
                datos[intencion] = round(float(peso) * decay, 6)
        for datos in self.memoria.get("frases", {}).values():
            for intencion, peso in list(datos.items()):
                if intencion.startswith("_"):
                    continue
                datos[intencion] = round(float(peso) * decay, 6)

    # ========================================================================
    # PESOS DE PALABRAS
    # ========================================================================

    def _obtener_palabra(self, palabra: str) -> Dict[str, float]:
        clave = clave_token(palabra)
        tabla = self.memoria.setdefault("palabras", {})
        if clave not in tabla:
            tabla[clave] = {}
        return tabla[clave]

    def peso_palabra(self, palabra: str, intencion: str) -> float:
        datos = self._obtener_palabra(palabra)
        return float(datos.get(normalizar_intencion(intencion), 0.0))

    def ajustar_peso_palabra(
        self,
        palabra: str,
        intencion: str,
        delta: float,
    ) -> float:
        datos = self._obtener_palabra(palabra)
        clave = normalizar_intencion(intencion)
        anterior = float(datos.get(clave, 0.0))
        incremento = delta * float(self._cfg("learning_rate", DEFAULT_LEARNING_RATE))
        nuevo = limitar(
            anterior + incremento,
            float(self._cfg("min_weight", DEFAULT_MIN_WEIGHT)),
            float(self._cfg("max_weight", DEFAULT_MAX_WEIGHT)),
        )
        datos[clave] = round(nuevo, 6)
        datos["_actualizado"] = ahora_iso()
        return datos[clave]

    def palabras_aprendidas(self, intencion: Optional[str] = None) -> Dict[str, Any]:
        """Devuelve palabras con peso aprendido."""
        resultado: Dict[str, Any] = {}
        objetivo = normalizar_intencion(intencion) if intencion else None
        for palabra, datos in self.memoria.get("palabras", {}).items():
            if objetivo is None:
                resultado[palabra] = deepcopy(datos)
            elif objetivo in datos:
                resultado[palabra] = {
                    objetivo: datos[objetivo]
                }
        return resultado

    # ========================================================================
    # PESOS DE FRASES
    # ========================================================================

    def _obtener_frase(self, frase: str) -> Dict[str, float]:
        clave = clave_frase(frase)
        tabla = self.memoria.setdefault("frases", {})
        if clave not in tabla:
            tabla[clave] = {}
        return tabla[clave]

    def peso_frase(self, frase: str, intencion: str) -> float:
        datos = self._obtener_frase(frase)
        return float(datos.get(normalizar_intencion(intencion), 0.0))

    def ajustar_peso_frase(
        self,
        frase: str,
        intencion: str,
        delta: float,
    ) -> float:
        datos = self._obtener_frase(frase)
        clave = normalizar_intencion(intencion)
        anterior = float(datos.get(clave, 0.0))
        incremento = delta * float(self._cfg("learning_rate", DEFAULT_LEARNING_RATE))
        nuevo = limitar(
            anterior + incremento,
            float(self._cfg("min_weight", DEFAULT_MIN_WEIGHT)),
            float(self._cfg("max_weight", DEFAULT_MAX_WEIGHT)),
        )
        datos[clave] = round(nuevo, 6)
        datos["_actualizado"] = ahora_iso()
        return datos[clave]

    def frases_aprendidas(self, intencion: Optional[str] = None) -> Dict[str, Any]:
        resultado: Dict[str, Any] = {}
        objetivo = normalizar_intencion(intencion) if intencion else None
        for frase, datos in self.memoria.get("frases", {}).items():
            if objetivo is None:
                resultado[frase] = deepcopy(datos)
            elif objetivo in datos:
                resultado[frase] = {
                    objetivo: datos[objetivo]
                }
        return resultado

    # ========================================================================
    # EJEMPLOS Y EPISODIOS
    # ========================================================================

    def _clave_ejemplo(self, mensaje: str, intencion: str) -> str:
        return f"{clave_frase(mensaje)}|||{normalizar_intencion(intencion)}"

    def registrar_ejemplo(
        self,
        mensaje: str,
        intencion: str,
        recompensa: float,
        acierto: bool,
    ) -> EjemploAprendido:
        clave = self._clave_ejemplo(mensaje, intencion)
        ejemplos = self.memoria.setdefault("ejemplos", {})
        datos = ejemplos.get(clave)
        if not isinstance(datos, dict):
            datos = {
                "mensaje": normalizar_mensaje(mensaje),
                "intencion": normalizar_intencion(intencion),
                "recompensa_total": 0.0,
                "apariciones": 0,
                "aciertos": 0,
                "rechazos": 0,
                "ultima_actualizacion": ahora_iso(),
            }
        datos["recompensa_total"] = float(datos.get("recompensa_total", 0.0)) + recompensa
        datos["apariciones"] = int(datos.get("apariciones", 0)) + 1
        if acierto:
            datos["aciertos"] = int(datos.get("aciertos", 0)) + 1
        else:
            datos["rechazos"] = int(datos.get("rechazos", 0)) + 1
        datos["ultima_actualizacion"] = ahora_iso()
        ejemplos[clave] = datos
        self._limitar_diccionario("ejemplos", int(self._cfg("max_examples", DEFAULT_MAX_EXAMPLES)))
        return EjemploAprendido(**datos)

    def buscar_ejemplos(
        self,
        mensaje: str,
        intencion: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        objetivo = normalizar_intencion(intencion) if intencion else None
        palabras = set(extraer_tokens_utiles(mensaje))
        resultados: List[Tuple[float, Dict[str, Any]]] = []
        for datos in self.memoria.get("ejemplos", {}).values():
            if objetivo and datos.get("intencion") != objetivo:
                continue
            ejemplo_tokens = set(extraer_tokens_utiles(datos.get("mensaje", "")))
            interseccion = len(palabras & ejemplo_tokens)
            union = max(len(palabras | ejemplo_tokens), 1)
            similitud_jaccard = interseccion / union
            recompensa = float(datos.get("recompensa_total", 0.0))
            puntuacion = similitud_jaccard + (sigmoide(recompensa) * 0.1)
            if puntuacion > 0:
                resultados.append((puntuacion, datos))
        resultados.sort(key=lambda x: x[0], reverse=True)
        return [deepcopy(datos) for _, datos in resultados[:10]]

    # ========================================================================
    # CORRECCIONES DE ESCRITURA
    # ========================================================================

    def registrar_correccion(
        self,
        original: str,
        corregida: str,
        confianza: float = 1.0,
    ) -> None:
        original_n = normalizar(original)
        corregida_n = normalizar(corregida)
        if not original_n or not corregida_n or original_n == corregida_n:
            return
        correcciones = self.memoria.setdefault("correcciones", {})
        datos = correcciones.get(original_n, {})
        datos[corregida_n] = float(datos.get(corregida_n, 0.0)) + max(confianza, 0.0)
        datos["_actualizado"] = ahora_iso()
        correcciones[original_n] = datos
        self._limitar_diccionario(
            "correcciones",
            int(self._cfg("max_corrections", DEFAULT_MAX_CORRECTIONS)),
        )

    def sugerir_correccion(self, palabra: str) -> Optional[str]:
        clave = normalizar(palabra)
        datos = self.memoria.get("correcciones", {}).get(clave, {})
        candidatos = [
            (valor, peso)
            for valor, peso in datos.items()
            if not valor.startswith("_")
        ]
        if not candidatos:
            return None
        candidatos.sort(key=lambda x: x[1], reverse=True)
        return candidatos[0][0]

    def correcciones_aprendidas(self) -> Dict[str, Any]:
        return deepcopy(self.memoria.get("correcciones", {}))

    # ========================================================================
    # ASOCIACIONES CONTEXTUALES
    # ========================================================================

    def _clave_asociacion(self, anterior: str, siguiente: str) -> str:
        return f"{normalizar_intencion(anterior)}->{normalizar_intencion(siguiente)}"

    def registrar_asociacion(
        self,
        anterior: str,
        siguiente: str,
        recompensa: float = 1.0,
    ) -> None:
        anterior_n = normalizar_intencion(anterior)
        siguiente_n = normalizar_intencion(siguiente)
        if not anterior_n or not siguiente_n:
            return
        clave = self._clave_asociacion(anterior_n, siguiente_n)
        tabla = self.memoria.setdefault("asociaciones", {})
        datos = tabla.get(clave)
        if not isinstance(datos, dict):
            datos = {
                "anterior": anterior_n,
                "siguiente": siguiente_n,
                "peso": 0.0,
                "apariciones": 0,
                "timestamp": ahora_iso(),
            }
        datos["peso"] = limitar(
            float(datos.get("peso", 0.0)) + recompensa * float(self._cfg("learning_rate", DEFAULT_LEARNING_RATE)),
            float(self._cfg("min_weight", DEFAULT_MIN_WEIGHT)),
            float(self._cfg("max_weight", DEFAULT_MAX_WEIGHT)),
        )
        datos["apariciones"] = int(datos.get("apariciones", 0)) + 1
        datos["timestamp"] = ahora_iso()
        tabla[clave] = datos
        self._limitar_diccionario(
            "asociaciones",
            int(self._cfg("max_associations", DEFAULT_MAX_ASSOCIATIONS)),
        )

    def asociaciones_desde(self, intencion: str) -> List[Dict[str, Any]]:
        objetivo = normalizar_intencion(intencion)
        resultados = []
        for datos in self.memoria.get("asociaciones", {}).values():
            if datos.get("anterior") == objetivo:
                resultados.append(deepcopy(datos))
        resultados.sort(key=lambda x: float(x.get("peso", 0.0)), reverse=True)
        return resultados

    # ========================================================================
    # RECOMPENSAS
    # ========================================================================

    def valor_recompensa(
        self,
        tipo: str,
        intensidad: str = "normal",
    ) -> float:
        tipo_n = normalizar(tipo)
        intensidad_n = normalizar(intensidad)
        if tipo_n in {"si", "sí", "confirmacion", "confirmar", "acierto", "correcto"}:
            if intensidad_n in {"fuerte", "strong", "alta", "alto"}:
                return float(self._cfg("strong_reward", DEFAULT_STRONG_REWARD))
            return float(self._cfg("reward", DEFAULT_REWARD))
        if tipo_n in {"no", "negacion", "negación", "rechazo", "rechazar", "error", "incorrecto"}:
            if intensidad_n in {"fuerte", "strong", "alta", "alto"}:
                return float(self._cfg("strong_penalty", DEFAULT_STRONG_PENALTY))
            return float(self._cfg("penalty", DEFAULT_PENALTY))
        return 0.0

    def recompensar(
        self,
        intencion: str,
        mensaje: str = "",
        intensidad: str = "normal",
        motivo: str = "confirmacion_usuario",
        evidencias: Optional[Sequence[str]] = None,
        confianza_previa: float = 0.0,
    ) -> AprendizajeResultado:
        valor = self.valor_recompensa("si", intensidad)
        return self._aplicar_recompensa(
            intencion=intencion,
            mensaje=mensaje,
            recompensa=valor,
            motivo=motivo,
            evidencias=evidencias,
            confianza_previa=confianza_previa,
            acierto=True,
        )

    def penalizar(
        self,
        intencion: str,
        mensaje: str = "",
        intensidad: str = "normal",
        motivo: str = "rechazo_usuario",
        evidencias: Optional[Sequence[str]] = None,
        confianza_previa: float = 0.0,
    ) -> AprendizajeResultado:
        valor = self.valor_recompensa("no", intensidad)
        return self._aplicar_recompensa(
            intencion=intencion,
            mensaje=mensaje,
            recompensa=valor,
            motivo=motivo,
            evidencias=evidencias,
            confianza_previa=confianza_previa,
            acierto=False,
        )

    def _aplicar_recompensa(
        self,
        intencion: str,
        mensaje: str,
        recompensa: float,
        motivo: str,
        evidencias: Optional[Sequence[str]],
        confianza_previa: float,
        acierto: bool,
    ) -> AprendizajeResultado:
        intencion_n = normalizar_intencion(intencion)
        datos = self._obtener_intencion(intencion_n)
        peso_anterior = float(datos.get("peso", 0.0))

        self.ajustar_peso_intencion(
            intencion_n,
            recompensa,
            motivo=motivo,
        )

        for palabra in extraer_tokens_utiles(mensaje):
            self.ajustar_peso_palabra(
                palabra,
                intencion_n,
                recompensa * 0.50,
            )

        if mensaje:
            self.ajustar_peso_frase(
                mensaje,
                intencion_n,
                recompensa * 0.75,
            )

        self.registrar_ejemplo(
            mensaje,
            intencion_n,
            recompensa,
            acierto=acierto,
        )

        datos = self._obtener_intencion(intencion_n)
        datos["apariciones"] = int(datos.get("apariciones", 0)) + 1
        datos["recompensa_total"] = float(datos.get("recompensa_total", 0.0)) + recompensa
        if acierto:
            datos["aciertos"] = int(datos.get("aciertos", 0)) + 1
            self.memoria["racha_aciertos"] = int(self.memoria.get("racha_aciertos", 0)) + 1
            self.memoria["racha_errores"] = 0
            self.memoria["mejor_racha"] = max(
                int(self.memoria.get("mejor_racha", 0)),
                int(self.memoria.get("racha_aciertos", 0)),
            )
        else:
            datos["errores"] = int(datos.get("errores", 0)) + 1
            self.memoria["racha_errores"] = int(self.memoria.get("racha_errores", 0)) + 1
            self.memoria["racha_aciertos"] = 0

        self.memoria["total_eventos"] = int(self.memoria.get("total_eventos", 0)) + 1
        if recompensa >= 0:
            self.memoria["total_recompensas"] = float(self.memoria.get("total_recompensas", 0.0)) + recompensa
        else:
            self.memoria["total_penalizaciones"] = float(self.memoria.get("total_penalizaciones", 0.0)) + abs(recompensa)

        peso_nuevo = float(datos.get("peso", 0.0))
        confianza_nueva = self.confianza_intencion(intencion_n)

        evento = AprendizajeResultado(
            mensaje=normalizar_mensaje(mensaje),
            intencion=intencion_n,
            recompensa=recompensa,
            peso_anterior=round(peso_anterior, 6),
            peso_nuevo=round(peso_nuevo, 6),
            motivo=motivo,
            timestamp=ahora_iso(),
            evento_id=generar_id(),
            confianza_previa=round(confianza_previa, 4),
            confianza_nueva=round(confianza_nueva, 4),
            evidencias=list(evidencias or []),
        )

        self._registrar_historial(evento)
        self._actualizar_sesion(mensaje, intencion_n, recompensa, evento.evento_id)
        self.guardar()
        return evento

    # ========================================================================
    # FEEDBACK DE TEXTO
    # ========================================================================

    def registrar_feedback(
        self,
        tipo: str,
        mensaje: str,
        intencion: str,
        contexto: Optional[Dict[str, Any]] = None,
        intensidad: str = "normal",
    ) -> AprendizajeResultado:
        """Registra un feedback explícito como sí/no."""
        tipo_n = normalizar(tipo)
        if tipo_n in {"si", "sí", "confirmar", "confirmacion", "correcto"}:
            resultado = self.recompensar(
                intencion=intencion,
                mensaje=mensaje,
                intensidad=intensidad,
                motivo="feedback_si",
                confianza_previa=float((contexto or {}).get("confianza", 0.0)),
            )
        elif tipo_n in {"no", "negar", "negacion", "incorrecto", "rechazo"}:
            resultado = self.penalizar(
                intencion=intencion,
                mensaje=mensaje,
                intensidad=intensidad,
                motivo="feedback_no",
                confianza_previa=float((contexto or {}).get("confianza", 0.0)),
            )
        else:
            raise ValueError("El tipo de feedback debe ser sí/si o no.")

        if contexto:
            self.memoria.setdefault("sesion", {})["ultimo_contexto"] = deepcopy(contexto)
            self.guardar()
        return resultado

    # ========================================================================
    # APRENDIZAJE DE RESULTADOS DEL MOTOR DE HIPÓTESIS
    # ========================================================================

    def aprender_de_resultados(
        self,
        mensaje: str,
        resultados: Sequence[Dict[str, Any]],
        intencion_correcta: Optional[str] = None,
        recompensa: Optional[float] = None,
        mantener_contexto: bool = True,
    ) -> List[AprendizajeResultado]:
        """Aprende comparando la intención correcta con sus competidoras."""
        if not resultados:
            return []

        objetivo = normalizar_intencion(intencion_correcta) if intencion_correcta else None
        resultados_n = [dict(item) for item in resultados]
        salida: List[AprendizajeResultado] = []

        if objetivo:
            valor = DEFAULT_REWARD if recompensa is None else recompensa
            principal = next(
                (r for r in resultados_n if normalizar_intencion(r.get("intencion", "")) == objetivo),
                None,
            )
            confianza = self.confianza_resultado(principal) if principal else 0.0
            salida.append(
                self._aplicar_recompensa(
                    intencion=objetivo,
                    mensaje=mensaje,
                    recompensa=valor,
                    motivo="intencion_correcta_confirmada",
                    evidencias=(principal or {}).get("evidencias", []),
                    confianza_previa=confianza,
                    acierto=True,
                )
            )

            for candidato in resultados_n:
                nombre = normalizar_intencion(candidato.get("intencion", ""))
                if not nombre or nombre == objetivo:
                    continue
                penalizacion = -abs(float(valor)) * 0.35
                confianza_candidato = self.confianza_resultado(candidato)
                salida.append(
                    self._aplicar_recompensa(
                        intencion=nombre,
                        mensaje=mensaje,
                        recompensa=penalizacion,
                        motivo="competidor_rechazado_por_feedback",
                        evidencias=candidato.get("evidencias", []),
                        confianza_previa=confianza_candidato,
                        acierto=False,
                    )
                )

            if mantener_contexto:
                self.memoria.setdefault("sesion", {})["ultima_intencion"] = objetivo
                self.guardar()
            return salida

        # Si no conocemos la intención correcta, premiamos el top si se envía una recompensa positiva.
        if recompensa is not None and resultados_n:
            top = resultados_n[0]
            salida.append(
                self._aplicar_recompensa(
                    intencion=normalizar_intencion(top.get("intencion", "")),
                    mensaje=mensaje,
                    recompensa=float(recompensa),
                    motivo="feedback_sobre_top",
                    evidencias=top.get("evidencias", []),
                    confianza_previa=self.confianza_resultado(top),
                    acierto=recompensa > 0,
                )
            )
        return salida

    def aprender_de_confirmacion(
        self,
        mensaje: str,
        resultados: Sequence[Dict[str, Any]],
        confirmada: bool,
    ) -> List[AprendizajeResultado]:
        """Atajo para el flujo sí/no de tamy_brain."""
        if not resultados:
            return []
        intencion = resultados[0].get("intencion")
        if not intencion:
            return []
        if confirmada:
            return self.aprender_de_resultados(
                mensaje,
                resultados,
                intencion_correcta=intencion,
            )
        return self.penalizar(
            intencion=normalizar_intencion(intencion),
            mensaje=mensaje,
            motivo="primera_hipotesis_rechazada",
            confianza_previa=self.confianza_resultado(resultados[0]),
        ) and []

    # ========================================================================
    # AJUSTE DIRECTO DE RESULTADOS
    # ========================================================================

    def bonificacion_resultado(
        self,
        resultado: Dict[str, Any],
    ) -> float:
        """Obtiene el bono aprendido para una hipótesis."""
        intencion = normalizar_intencion(resultado.get("intencion", ""))
        if not intencion:
            return 0.0

        bono = self.peso_intencion(intencion)
        evidencias = resultado.get("evidencias", []) or []
        for evidencia in evidencias:
            if evidencia.startswith("palabra:"):
                palabra = evidencia.split(":", 1)[1].strip()
                bono += self.peso_palabra(palabra, intencion)
            elif evidencia.startswith("frase:"):
                frase = evidencia.split(":", 1)[1].strip()
                bono += self.peso_frase(frase, intencion)
        ejemplos = self.buscar_ejemplos(
            str(self.memoria.get("sesion", {}).get("ultimo_mensaje", "")),
            intencion=intencion,
        )
        if ejemplos:
            bono += min(len(ejemplos) * 0.05, 0.50)
        return bono

    def ajustar_resultados(
        self,
        mensaje: str,
        resultados: Sequence[Dict[str, Any]],
        contexto: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Suma aprendizaje al score original y devuelve una copia ordenada."""
        salida: List[Dict[str, Any]] = []
        for item in resultados:
            datos = dict(item)
            base = float(datos.get("puntuacion", 0.0))
            intencion = normalizar_intencion(datos.get("intencion", ""))
            datos["puntuacion_base"] = round(base, 6)
            datos["bono_aprendizaje"] = round(self._bono_para_mensaje(mensaje, intencion), 6)
            datos["puntuacion"] = round(base + datos["bono_aprendizaje"], 6)
            datos["confianza"] = round(self.confianza_resultado(datos), 4)
            datos["nivel_confianza"] = self.etiqueta_confianza(datos["confianza"])
            salida.append(datos)

        salida.sort(key=lambda x: float(x.get("puntuacion", 0.0)), reverse=True)
        if contexto:
            self.aplicar_contexto(salida, contexto)
        return salida

    def _bono_para_mensaje(self, mensaje: str, intencion: str) -> float:
        if not intencion:
            return 0.0
        bono = self.peso_intencion(intencion)
        for palabra in extraer_tokens_utiles(mensaje):
            bono += self.peso_palabra(palabra, intencion)
        clave_mensaje = clave_frase(mensaje)
        if clave_mensaje:
            bono += self.peso_frase(clave_mensaje, intencion)
        return bono

    # ========================================================================
    # CONFIANZA
    # ========================================================================

    def confianza_intencion(self, intencion: str) -> float:
        datos = self._obtener_intencion(intencion)
        recompensa = float(datos.get("recompensa_total", 0.0))
        aciertos = int(datos.get("aciertos", 0))
        errores = int(datos.get("errores", 0))
        total = aciertos + errores
        tasa = aciertos / total if total else 0.0
        evidencia = sigmoide(recompensa / 4.0)
        experiencia = min(total / 10.0, 1.0)
        confianza = 0.45 * tasa + 0.35 * evidencia + 0.20 * experiencia
        return limitar(confianza, 0.0, 1.0)

    def confianza_resultado(self, resultado: Optional[Dict[str, Any]]) -> float:
        if not resultado:
            return 0.0
        puntuacion = float(resultado.get("puntuacion", 0.0))
        aprendizaje = float(resultado.get("bono_aprendizaje", 0.0))
        total = max(abs(puntuacion) + abs(aprendizaje), 1.0)
        base = sigmoide(total / 4.0 - 1.0)
        intencion = resultado.get("intencion")
        experiencia = self.confianza_intencion(intencion) if intencion else 0.0
        return round(limitar(0.70 * base + 0.30 * experiencia, 0.0, 1.0), 4)

    def etiqueta_confianza(self, confianza: float) -> str:
        confianza = limitar(float(confianza), 0.0, 1.0)
        if confianza >= 0.85:
            return "muy alta"
        if confianza >= 0.65:
            return "alta"
        if confianza >= 0.45:
            return "media"
        if confianza >= 0.25:
            return "baja"
        return "muy baja"

    # ========================================================================
    # CONTEXTO Y SESIÓN
    # ========================================================================

    def _actualizar_sesion(
        self,
        mensaje: str,
        intencion: str,
        recompensa: float,
        evento_id: str,
    ) -> None:
        sesion = self.memoria.setdefault("sesion", {})
        sesion["ultimo_mensaje"] = normalizar_mensaje(mensaje)
        sesion["ultima_intencion"] = normalizar_intencion(intencion)
        sesion["ultima_recompensa"] = recompensa
        sesion["ultimo_evento"] = evento_id

    def establecer_pregunta(self, pregunta: str) -> None:
        self.memoria.setdefault("sesion", {})["ultima_pregunta"] = str(pregunta or "")
        self.guardar()

    def contexto_actual(self) -> Dict[str, Any]:
        return deepcopy(self.memoria.get("sesion", {}))

    def aplicar_contexto(
        self,
        resultados: Sequence[Dict[str, Any]],
        contexto: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not contexto:
            return list(resultados)
        ultima = contexto.get("ultima_intencion")
        if not ultima:
            return list(resultados)
        salida = [dict(item) for item in resultados]
        for item in salida:
            if normalizar_intencion(item.get("intencion", "")) == normalizar_intencion(ultima):
                item["puntuacion"] = round(float(item.get("puntuacion", 0.0)) + 0.75, 6)
                item.setdefault("evidencias", []).append("aprendizaje:contexto_anterior")
        salida.sort(key=lambda x: float(x.get("puntuacion", 0.0)), reverse=True)
        return salida

    def registrar_transicion(
        self,
        anterior: Optional[str],
        siguiente: Optional[str],
        recompensa: float = 1.0,
    ) -> None:
        if not anterior or not siguiente:
            return
        self.registrar_asociacion(anterior, siguiente, recompensa=recompensa)
        self.guardar()

    # ========================================================================
    # ESTADÍSTICAS
    # ========================================================================

    def estadisticas_generales(self) -> Dict[str, Any]:
        memoria = self.memoria
        total = int(memoria.get("total_eventos", 0))
        recompensas = float(memoria.get("total_recompensas", 0.0))
        penalizaciones = float(memoria.get("total_penalizaciones", 0.0))
        neto = recompensas - penalizaciones
        return {
            "version": VERSION,
            "eventos": total,
            "recompensas": round(recompensas, 6),
            "penalizaciones": round(penalizaciones, 6),
            "neto": round(neto, 6),
            "racha_actual": int(memoria.get("racha_aciertos", 0)),
            "mejor_racha": int(memoria.get("mejor_racha", 0)),
            "intenciones_aprendidas": len(memoria.get("intenciones", {})),
            "palabras_aprendidas": len(memoria.get("palabras", {})),
            "frases_aprendidas": len(memoria.get("frases", {})),
            "ejemplos": len(memoria.get("ejemplos", {})),
            "correcciones": len(memoria.get("correcciones", {})),
            "asociaciones": len(memoria.get("asociaciones", {})),
        }

    def estadisticas_intencion(self, intencion: str) -> Dict[str, Any]:
        datos = self._obtener_intencion(intencion)
        aciertos = int(datos.get("aciertos", 0))
        errores = int(datos.get("errores", 0))
        total = aciertos + errores
        return {
            "intencion": normalizar_intencion(intencion),
            "peso": float(datos.get("peso", 0.0)),
            "aciertos": aciertos,
            "errores": errores,
            "apariciones": int(datos.get("apariciones", 0)),
            "recompensa_total": float(datos.get("recompensa_total", 0.0)),
            "tasa_acierto": round(aciertos / total, 4) if total else 0.0,
            "confianza": round(self.confianza_intencion(intencion), 4),
            "ultima_actualizacion": datos.get("ultima_actualizacion"),
        }

    def ranking_intenciones(self, limite: int = 10) -> List[Dict[str, Any]]:
        nombres = list(self.memoria.get("intenciones", {}).keys())
        resultados = [self.estadisticas_intencion(nombre) for nombre in nombres]
        resultados.sort(
            key=lambda x: (x["confianza"], x["peso"], x["tasa_acierto"]),
            reverse=True,
        )
        return resultados[:limite]

    def top_palabras(self, limite: int = 20) -> List[Dict[str, Any]]:
        filas = []
        for palabra, datos in self.memoria.get("palabras", {}).items():
            for intencion, peso in datos.items():
                if intencion.startswith("_"):
                    continue
                filas.append({
                    "palabra": palabra,
                    "intencion": intencion,
                    "peso": round(float(peso), 6),
                })
        filas.sort(key=lambda x: abs(x["peso"]), reverse=True)
        return filas[:limite]

    def top_frases(self, limite: int = 20) -> List[Dict[str, Any]]:
        filas = []
        for frase, datos in self.memoria.get("frases", {}).items():
            for intencion, peso in datos.items():
                if intencion.startswith("_"):
                    continue
                filas.append({
                    "frase": frase,
                    "intencion": intencion,
                    "peso": round(float(peso), 6),
                })
        filas.sort(key=lambda x: abs(x["peso"]), reverse=True)
        return filas[:limite]

    # ========================================================================
    # HISTORIAL
    # ========================================================================

    def _registrar_historial(self, evento: AprendizajeResultado) -> None:
        historial = self.memoria.setdefault("historial", [])
        historial.append(evento.to_dict())
        maximo = int(self._cfg("max_history", DEFAULT_MAX_HISTORY))
        if len(historial) > maximo:
            del historial[:-maximo]

    def historial(self, limite: int = 50) -> List[Dict[str, Any]]:
        datos = self.memoria.get("historial", [])
        return deepcopy(datos[-max(limite, 0):])

    def ultimo_evento(self) -> Optional[Dict[str, Any]]:
        historial = self.memoria.get("historial", [])
        if not historial:
            return None
        return deepcopy(historial[-1])

    # ========================================================================
    # LIMPIEZA Y LÍMITES
    # ========================================================================

    def _limitar_diccionario(self, nombre: str, limite: int) -> None:
        tabla = self.memoria.get(nombre, {})
        if not isinstance(tabla, dict):
            self.memoria[nombre] = {}
            return
        if len(tabla) <= limite:
            return

        def antiguedad(item: Tuple[str, Any]) -> str:
            datos = item[1]
            if isinstance(datos, dict):
                return str(
                    datos.get("_actualizado")
                    or datos.get("ultima_actualizacion")
                    or datos.get("timestamp")
                    or ""
                )
            return ""

        ordenado = sorted(tabla.items(), key=antiguedad, reverse=True)
        self.memoria[nombre] = dict(ordenado[:limite])

    def limpiar_historial(self) -> None:
        self.memoria["historial"] = []
        self.guardar()

    def limpiar_ejemplos(self) -> None:
        self.memoria["ejemplos"] = {}
        self.guardar()

    def limpiar_palabras(self) -> None:
        self.memoria["palabras"] = {}
        self.guardar()

    def limpiar_frases(self) -> None:
        self.memoria["frases"] = {}
        self.guardar()

    def limpiar_asociaciones(self) -> None:
        self.memoria["asociaciones"] = {}
        self.guardar()

    def limpiar_correcciones(self) -> None:
        self.memoria["correcciones"] = {}
        self.guardar()

    def reiniciar_aprendizaje(self) -> None:
        """Borra pesos y estadísticas aprendidas pero conserva configuración."""
        config = deepcopy(self.memoria.get("config", {}))
        sesion = deepcopy(self.memoria.get("sesion", {}))
        self.memoria = estado_vacio()
        self.memoria["config"] = config
        self.memoria["sesion"] = sesion
        self.guardar()

    # ========================================================================
    # EXPORTACIÓN / IMPORTACIÓN
    # ========================================================================

    def exportar(self, ruta: os.PathLike[str] | str) -> bool:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        try:
            with ruta.open("w", encoding="utf-8") as archivo:
                json.dump(
                    self.memoria,
                    archivo,
                    indent=4,
                    ensure_ascii=False,
                )
            return True
        except OSError:
            return False

    def importar(self, ruta: os.PathLike[str] | str, reemplazar: bool = True) -> bool:
        ruta = Path(ruta)
        try:
            with ruta.open("r", encoding="utf-8") as archivo:
                datos = json.load(archivo)
        except (OSError, json.JSONDecodeError):
            return False

        if reemplazar:
            self.memoria = self._fusionar_con_estado_base(datos)
        else:
            self.memoria = self._fusionar_profundo(self.memoria, datos)
        self.guardar()
        return True

    def _fusionar_profundo(self, destino: Dict[str, Any], fuente: Dict[str, Any]) -> Dict[str, Any]:
        resultado = deepcopy(destino)
        for clave, valor in fuente.items():
            if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
                resultado[clave] = self._fusionar_profundo(resultado[clave], valor)
            else:
                resultado[clave] = deepcopy(valor)
        return resultado

    # ========================================================================
    # APRENDIZAJE DE REGLAS EXPLÍCITAS
    # ========================================================================

    def enseñar_palabra(self, palabra: str, intencion: str, valor: float = 2.0) -> float:
        """Enseña manualmente que una palabra apunta a una intención."""
        resultado = self.ajustar_peso_palabra(palabra, intencion, valor)
        self.guardar()
        return resultado

    def enseñar_frase(self, frase: str, intencion: str, valor: float = 3.0) -> float:
        """Enseña manualmente que una frase apunta a una intención."""
        resultado = self.ajustar_peso_frase(frase, intencion, valor)
        self.guardar()
        return resultado

    def enseñar_correccion(self, original: str, corregida: str, confianza: float = 1.0) -> None:
        """Enseña manualmente una corrección ortográfica."""
        self.registrar_correccion(original, corregida, confianza)
        self.guardar()

    def enseñar_asociacion(self, anterior: str, siguiente: str, valor: float = 2.0) -> None:
        """Enseña manualmente una transición contextual."""
        self.registrar_asociacion(anterior, siguiente, recompensa=valor)
        self.guardar()

    # ========================================================================
    # REGLAS DE RECOMPENSA ADAPTATIVA
    # ========================================================================

    def recompensa_adaptativa(self, confianza: float, confirmado: bool) -> float:
        """Aumenta o reduce la intensidad del aprendizaje según confianza."""
        confianza = limitar(float(confianza), 0.0, 1.0)
        if confirmado:
            minimo = 0.50
            factor = minimo + confianza
            return round(float(self._cfg("reward", DEFAULT_REWARD)) * factor, 6)
        minimo = 0.50
        factor = minimo + confianza
        return round(float(self._cfg("penalty", DEFAULT_PENALTY)) * factor, 6)

    def premio_por_rango(self, posicion: int) -> float:
        """Define cuánto premio corresponde por posición de hipótesis."""
        tabla = {
            1: 1.0,
            2: 0.5,
            3: 0.25,
            4: 0.10,
        }
        return tabla.get(int(posicion), 0.0)

    def castigo_por_rango(self, posicion: int) -> float:
        """Define cuánto castigo recibe una hipótesis rechazada."""
        tabla = {
            1: -1.0,
            2: -0.50,
            3: -0.25,
            4: -0.10,
        }
        return tabla.get(int(posicion), 0.0)

    # ========================================================================
    # EXPLORACIÓN / EXPLOTACIÓN
    # ========================================================================

    def probabilidad_exploracion(self) -> float:
        return limitar(float(self._cfg("exploration", DEFAULT_EXPLORATION)), 0.0, 1.0)

    def deberia_explorar(self, semilla: Optional[float] = None) -> bool:
        """Decisión pseudoaleatoria simple para favorecer exploración ocasional."""
        if semilla is None:
            semilla = (time.time() % 1.0)
        return float(semilla) < self.probabilidad_exploracion()

    # ========================================================================
    # SELECCIÓN DE HIPÓTESIS
    # ========================================================================

    def seleccionar_hipotesis(
        self,
        resultados: Sequence[Dict[str, Any]],
        exploracion: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Selecciona una hipótesis aprendida sin romper el orden original."""
        if not resultados:
            return None
        ajustados = [dict(item) for item in resultados]
        for item in ajustados:
            item["bono_aprendizaje"] = self._bono_para_mensaje(
                str(self.memoria.get("sesion", {}).get("ultimo_mensaje", "")),
                str(item.get("intencion", "")),
            )
            item["puntuacion_aprendida"] = float(item.get("puntuacion", 0.0)) + float(item["bono_aprendizaje"])
        ajustados.sort(key=lambda x: float(x.get("puntuacion_aprendida", 0.0)), reverse=True)
        if exploracion and len(ajustados) > 1:
            # Exploración determinista mínima: intercambia primero/segundo solo si el margen es pequeño.
            margen = float(ajustados[0]["puntuacion_aprendida"]) - float(ajustados[1]["puntuacion_aprendida"])
            if margen < float(self._cfg("confirmation_gap", DEFAULT_CONFIRMATION_GAP)) / 2:
                return ajustados[1]
        return ajustados[0]

    def reordenar_por_aprendizaje(
        self,
        mensaje: str,
        resultados: Sequence[Dict[str, Any]],
        contexto: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Aplica pesos aprendidos a las hipótesis de una consulta."""
        salida = self.ajustar_resultados(mensaje, resultados, contexto=contexto)
        for item in salida:
            item["mensaje_aprendizaje"] = mensaje
        return salida

    # ========================================================================
    # CONFIRMACIÓN Y DESAMBIGUACIÓN
    # ========================================================================

    def necesita_confirmacion(
        self,
        resultados: Sequence[Dict[str, Any]],
    ) -> bool:
        if not resultados:
            return True
        if len(resultados) == 1:
            return bool(resultados[0].get("requiere_confirmacion", False))
        top = float(resultados[0].get("puntuacion", 0.0))
        segundo = float(resultados[1].get("puntuacion", 0.0))
        margen = top - segundo
        return margen <= float(self._cfg("confirmation_gap", DEFAULT_CONFIRMATION_GAP))

    def diferencia_mejores(self, resultados: Sequence[Dict[str, Any]]) -> float:
        if len(resultados) < 2:
            return float("inf")
        return float(resultados[0].get("puntuacion", 0.0)) - float(resultados[1].get("puntuacion", 0.0))

    def resumen_hipotesis(self, resultados: Sequence[Dict[str, Any]], limite: int = 3) -> List[str]:
        lineas = []
        for posicion, item in enumerate(resultados[:limite], start=1):
            confianza = self.confianza_resultado(item)
            lineas.append(
                f"{posicion}. {item.get('intencion')} -> {float(item.get('puntuacion', 0.0)):.3f} | {confianza:.0%}"
            )
        return lineas

    # ========================================================================
    # APRENDIZAJE POR COMPARACIÓN
    # ========================================================================

    def comparar_hipotesis(
        self,
        primera: Dict[str, Any],
        segunda: Dict[str, Any],
    ) -> Dict[str, Any]:
        p1 = float(primera.get("puntuacion", 0.0))
        p2 = float(segunda.get("puntuacion", 0.0))
        diferencia = p1 - p2
        return {
            "mejor": primera.get("intencion") if diferencia >= 0 else segunda.get("intencion"),
            "peor": segunda.get("intencion") if diferencia >= 0 else primera.get("intencion"),
            "diferencia": round(abs(diferencia), 6),
            "ambigua": abs(diferencia) <= float(self._cfg("confirmation_gap", DEFAULT_CONFIRMATION_GAP)),
        }

    def penalizar_competidores(
        self,
        resultados: Sequence[Dict[str, Any]],
        intencion_correcta: str,
        intensidad: float = 0.50,
    ) -> None:
        objetivo = normalizar_intencion(intencion_correcta)
        for item in resultados:
            nombre = normalizar_intencion(item.get("intencion", ""))
            if not nombre or nombre == objetivo:
                continue
            self.ajustar_peso_intencion(
                nombre,
                -abs(intensidad),
                motivo="competidor",
            )
        self.guardar()

    # ========================================================================
    # APRENDIZAJE SEMÁNTICO SIMPLE
    # ========================================================================

    def aprender_relacion_token(
        self,
        mensaje: str,
        intencion: str,
        recompensa: float,
    ) -> Dict[str, float]:
        cambios = {}
        for palabra in extraer_tokens_utiles(mensaje):
            cambios[palabra] = self.ajustar_peso_palabra(
                palabra,
                intencion,
                recompensa,
            )
        self.guardar()
        return cambios

    def aprender_relacion_frase(
        self,
        mensaje: str,
        intencion: str,
        recompensa: float,
    ) -> float:
        resultado = self.ajustar_peso_frase(
            mensaje,
            intencion,
            recompensa,
        )
        self.guardar()
        return resultado

    def aprender_relacion_completa(
        self,
        mensaje: str,
        intencion: str,
        recompensa: float,
    ) -> AprendizajeResultado:
        return self._aplicar_recompensa(
            intencion=intencion,
            mensaje=mensaje,
            recompensa=recompensa,
            motivo="aprendizaje_completo",
            evidencias=[],
            confianza_previa=0.0,
            acierto=recompensa > 0,
        )

    # ========================================================================
    # DETECCIÓN DE CONOCIMIENTO ADQUIRIDO
    # ========================================================================

    def conocimiento_adquirido(self, minimo_peso: float = 1.0) -> List[Dict[str, Any]]:
        resultado = []
        for palabra, datos in self.memoria.get("palabras", {}).items():
            for intencion, peso in datos.items():
                if intencion.startswith("_"):
                    continue
                if abs(float(peso)) >= minimo_peso:
                    resultado.append({
                        "palabra": palabra,
                        "intencion": intencion,
                        "peso": round(float(peso), 6),
                    })
        resultado.sort(key=lambda x: abs(x["peso"]), reverse=True)
        return resultado

    def conocimiento_por_intencion(self, intencion: str, minimo_peso: float = 0.1) -> Dict[str, Any]:
        objetivo = normalizar_intencion(intencion)
        palabras = []
        for palabra, datos in self.memoria.get("palabras", {}).items():
            peso = float(datos.get(objetivo, 0.0))
            if abs(peso) >= minimo_peso:
                palabras.append((palabra, peso))
        palabras.sort(key=lambda x: abs(x[1]), reverse=True)
        return {
            "intencion": objetivo,
            "peso_global": self.peso_intencion(objetivo),
            "palabras": [
                {"palabra": palabra, "peso": round(peso, 6)}
                for palabra, peso in palabras
            ],
            "frases": self.frases_aprendidas(objetivo),
            "confianza": self.confianza_intencion(objetivo),
        }

    # ========================================================================
    # VALIDACIÓN DEL ESTADO
    # ========================================================================

    def validar_memoria(self) -> List[str]:
        errores: List[str] = []
        requerido = [
            "schema_version",
            "version_motor",
            "intenciones",
            "palabras",
            "frases",
            "ejemplos",
            "correcciones",
            "asociaciones",
            "historial",
            "sesion",
            "config",
        ]
        for clave in requerido:
            if clave not in self.memoria:
                errores.append(f"falta:{clave}")
        if not isinstance(self.memoria.get("historial"), list):
            errores.append("historial_no_es_lista")
        for nombre in ("intenciones", "palabras", "frases", "ejemplos", "correcciones", "asociaciones"):
            if not isinstance(self.memoria.get(nombre), dict):
                errores.append(f"{nombre}_no_es_diccionario")
        return errores

    def corregir_memoria(self) -> List[str]:
        errores = self.validar_memoria()
        self.memoria = self._fusionar_con_estado_base(self.memoria)
        self.guardar()
        return errores

    # ========================================================================
    # SNAPSHOT / RESTAURACIÓN
    # ========================================================================

    def snapshot(self) -> Dict[str, Any]:
        return deepcopy(self.memoria)

    def restaurar_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        if not isinstance(snapshot, dict):
            return False
        self.memoria = self._fusionar_con_estado_base(snapshot)
        return self.guardar()

    # ========================================================================
    # COMANDOS HUMANOS DE ENTRENAMIENTO
    # ========================================================================

    def entrenar(
        self,
        mensaje: str,
        intencion: str,
        correcto: bool = True,
        intensidad: str = "normal",
    ) -> AprendizajeResultado:
        """Entrena directamente con un mensaje y su intención correcta."""
        if correcto:
            return self.recompensar(
                intencion=intencion,
                mensaje=mensaje,
                intensidad=intensidad,
                motivo="entrenamiento_manual",
            )
        return self.penalizar(
            intencion=intencion,
            mensaje=mensaje,
            intensidad=intensidad,
            motivo="entrenamiento_negativo",
        )

    def entrenar_par(
        self,
        mensaje: str,
        correcta: str,
        incorrecta: Optional[str] = None,
        intensidad: str = "normal",
    ) -> List[AprendizajeResultado]:
        resultados = [
            self.recompensar(
                correcta,
                mensaje=mensaje,
                intensidad=intensidad,
                motivo="entrenamiento_par_correcto",
            )
        ]
        if incorrecta:
            resultados.append(
                self.penalizar(
                    incorrecta,
                    mensaje=mensaje,
                    intensidad=intensidad,
                    motivo="entrenamiento_par_incorrecto",
                )
            )
        return resultados

    def entrenar_frases(
        self,
        frases: Sequence[str],
        intencion: str,
        intensidad: str = "normal",
    ) -> List[AprendizajeResultado]:
        resultados = []
        for frase in frases:
            resultados.append(
                self.recompensar(
                    intencion,
                    mensaje=frase,
                    intensidad=intensidad,
                    motivo="entrenamiento_lote",
                )
            )
        return resultados

    # ========================================================================
    # INTERFAZ DE DIAGNÓSTICO
    # ========================================================================

    def diagnostico(self) -> Dict[str, Any]:
        return {
            "estadisticas": self.estadisticas_generales(),
            "config": self.config(),
            "validacion": self.validar_memoria(),
            "ranking": self.ranking_intenciones(10),
            "top_palabras": self.top_palabras(10),
            "top_frases": self.top_frases(10),
            "sesion": self.contexto_actual(),
        }

    def resumen_texto(self) -> str:
        est = self.estadisticas_generales()
        lineas = [
            "=== TAMY / MOTOR DE APRENDIZAJE ===",
            f"Eventos: {est['eventos']}",
            f"Recompensas: {est['recompensas']:.3f}",
            f"Penalizaciones: {est['penalizaciones']:.3f}",
            f"Neto: {est['neto']:.3f}",
            f"Mejor racha: {est['mejor_racha']}",
            f"Intenciones aprendidas: {est['intenciones_aprendidas']}",
            f"Palabras aprendidas: {est['palabras_aprendidas']}",
            f"Frases aprendidas: {est['frases_aprendidas']}",
            f"Ejemplos: {est['ejemplos']}",
            f"Correcciones: {est['correcciones']}",
            f"Asociaciones: {est['asociaciones']}",
        ]
        return "\n".join(lineas)

    # ========================================================================
    # API DE COMPATIBILIDAD / ATAJOS
    # ========================================================================

    def get_weight(self, intencion: str) -> float:
        return self.peso_intencion(intencion)

    def set_weight(self, intencion: str, valor: float) -> float:
        actual = self.peso_intencion(intencion)
        delta = float(valor) - actual
        resultado = self.ajustar_peso_intencion(intencion, delta, motivo="set_weight")
        self.guardar()
        return resultado

    def reward(self, intencion: str, mensaje: str = "") -> AprendizajeResultado:
        return self.recompensar(intencion, mensaje=mensaje)

    def punish(self, intencion: str, mensaje: str = "") -> AprendizajeResultado:
        return self.penalizar(intencion, mensaje=mensaje)

    def learn(self, mensaje: str, intencion: str, correcto: bool = True) -> AprendizajeResultado:
        return self.entrenar(mensaje, intencion, correcto=correcto)

    def save(self) -> bool:
        return self.guardar()

    def load(self) -> Dict[str, Any]:
        return self.cargar()


# ============================================================================
# FUNCIONES DE MÓDULO
# ============================================================================


_motor_global: Optional[MotorAprendizaje] = None


def crear_motor(
    ruta_memoria: os.PathLike[str] | str = DEFAULT_MEMORY_PATH,
) -> MotorAprendizaje:
    return MotorAprendizaje(ruta_memoria=ruta_memoria)


def obtener_motor(
    ruta_memoria: os.PathLike[str] | str = DEFAULT_MEMORY_PATH,
) -> MotorAprendizaje:
    global _motor_global
    if _motor_global is None:
        _motor_global = MotorAprendizaje(ruta_memoria=ruta_memoria)
    return _motor_global


def recompensar(
    intencion: str,
    mensaje: str = "",
    intensidad: str = "normal",
) -> AprendizajeResultado:
    return obtener_motor().recompensar(
        intencion=intencion,
        mensaje=mensaje,
        intensidad=intensidad,
    )


def penalizar(
    intencion: str,
    mensaje: str = "",
    intensidad: str = "normal",
) -> AprendizajeResultado:
    return obtener_motor().penalizar(
        intencion=intencion,
        mensaje=mensaje,
        intensidad=intensidad,
    )


def registrar_feedback(
    tipo: str,
    mensaje: str,
    intencion: str,
    contexto: Optional[Dict[str, Any]] = None,
) -> AprendizajeResultado:
    return obtener_motor().registrar_feedback(
        tipo=tipo,
        mensaje=mensaje,
        intencion=intencion,
        contexto=contexto,
    )


def ajustar_resultados(
    mensaje: str,
    resultados: Sequence[Dict[str, Any]],
    contexto: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return obtener_motor().ajustar_resultados(
        mensaje,
        resultados,
        contexto=contexto,
    )


def aprender(
    mensaje: str,
    intencion: str,
    correcto: bool = True,
) -> AprendizajeResultado:
    return obtener_motor().entrenar(
        mensaje,
        intencion,
        correcto=correcto,
    )


def diagnostico() -> Dict[str, Any]:
    return obtener_motor().diagnostico()


# ============================================================================
# DATOS DE ENTRENAMIENTO INICIALES OPCIONALES
# ============================================================================


ENTRENAMIENTO_BASICO: Tuple[Tuple[str, str], ...] = (
    ("hola", "saludar"),
    ("buenas", "saludar"),
    ("qué tal", "saludar"),
    ("quién eres", "identidad"),
    ("cómo te llamas", "identidad"),
    ("qué puedes hacer", "capacidades"),
    ("necesito ayuda", "ayuda"),
    ("qué hora es", "hora"),
    ("qué fecha es", "fecha"),
    ("qué mes es", "mes"),
    ("en qué año estamos", "año"),
    ("qué recuerdas", "memoria"),
    ("qué te gusta", "gustos"),
    ("quiero que aprendas", "aprendizaje"),
    ("quiero enseñarte", "enseñar"),
    ("quiero que razones", "razonamiento"),
    ("quiero trabajar", "trabajar"),
    ("necesito trabajar", "trabajar"),
    ("vamos a trabajar", "trabajar"),
    ("quiero estudiar", "estudiar"),
    ("tengo una tarea", "tareas"),
    ("qué tareas tengo", "tareas"),
    ("quiero buscar un archivo", "buscar_archivo"),
    ("abre un archivo", "abrir_archivo"),
    ("crea un archivo", "crear_archivo"),
    ("recuérdame esto", "guardar_memoria"),
    ("olvida esto", "borrar_memoria"),
)


CORRECCIONES_BÁSICAS: Tuple[Tuple[str, str], ...] = (
    ("tranbajar", "trabajar"),
    ("trabajr", "trabajar"),
    ("trbajar", "trabajar"),
    ("proycto", "proyecto"),
    ("qiero", "quiero"),
    ("kiero", "quiero"),
    ("aprenderas", "aprenderás"),
    ("ayudame", "ayúdame"),
    ("aplicacion", "aplicación"),
)


# ============================================================================
# ENTRENAMIENTO AUTOMÁTICO CONTROLADO
# ============================================================================
# API FINAL / ENTRENAMIENTO MANUAL
# ============================================================================
def entrenar_base(motor: Optional[MotorAprendizaje] = None) -> int:
    motor = motor or obtener_motor()
    total = 0
    for mensaje, intencion in ENTRENAMIENTO_BASICO:
        motor.entrenar(mensaje, intencion, correcto=True)
        total += 1
    for original, corregida in CORRECCIONES_BÁSICAS:
        motor.enseñar_correccion(original, corregida)
    motor.guardar()
    return total


def formatear_aprendizaje(resultado: AprendizajeResultado) -> str:
    signo = "+" if resultado.recompensa >= 0 else ""
    return (f"🧠 {resultado.intencion} "
            f"{resultado.peso_anterior:.3f} -> {resultado.peso_nuevo:.3f} "
            f"({signo}{resultado.recompensa:.3f}) [{resultado.motivo}]")


def ejecutar_pruebas() -> None:
    motor = MotorAprendizaje(Path("/tmp/tamy_learning_demo.json"))
    motor.entrenar("quiero trabajar", "trabajar", correcto=True)
    motor.entrenar("necesito trabajar", "trabajar", correcto=True)
    motor.entrenar("quiero estudiar", "trabajar", correcto=False)
    print(motor.resumen_texto())
    print(json.dumps(motor.conocimiento_por_intencion("trabajar"), ensure_ascii=False, indent=4))

__all__ = [
    "AprendizajeResultado", "Feedback", "EjemploAprendido", "AsociacionContextual",
    "PoliticaAprendizaje", "PoliticaAdaptativa", "MotorAprendizaje",
    "crear_motor", "obtener_motor", "recompensar", "penalizar",
    "registrar_feedback", "ajustar_resultados", "aprender", "diagnostico",
    "entrenar_lote", "entrenar_con_rechazo", "inspeccionar_aprendizaje",
    "entrenar_base", "formatear_aprendizaje", "mantenimiento", "comprimir_memoria",
]

if __name__ == "__main__":
    ejecutar_pruebas()
