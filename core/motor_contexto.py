
from __future__ import annotations

import json
import os
import re
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from .motor_hipotesis import normalizar, tokens
except ImportError:
    def normalizar(texto: str) -> str:
        texto = str(texto or "").lower().strip()
        texto = re.sub(r"[¿?!¡.,;:]+", " ", texto)
        texto = re.sub(r"\s+", " ", texto)
        return texto.strip()

    def tokens(texto: str) -> List[str]:
        return normalizar(texto).split()


VERSION = "1.0"
SCHEMA_VERSION = 1
DEFAULT_PATH = Path("data/tamy_context.json")
MAX_HISTORY = 30
MAX_FACTS = 100
MAX_REFERENTS = 50
MAX_EVENTS = 200
MAX_TOPICS = 20


@dataclass
class Turno:
    id: str
    usuario: str
    respuesta: str = ""
    intencion: Optional[str] = None
    confianza: float = 0.0
    timestamp: str = ""
    tema: Optional[str] = None
    pregunta_pendiente: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreguntaPendiente:
    texto: str
    tipo_respuesta: str = "libre"
    intencion: Optional[str] = None
    opciones: List[str] = field(default_factory=list)
    creada_en: str = ""
    activa: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Referente:
    clave: str
    valor: str
    tipo: str = "objeto"
    fuente: str = "conversacion"
    confianza: float = 1.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventoContextual:
    tipo: str
    mensaje: str
    valor: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ahora_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def nuevo_id(prefijo: str = "ctx") -> str:
    return f"{prefijo}_{uuid.uuid4().hex[:12]}"


def limitar(valor: float, minimo: float = 0.0, maximo: float = 1.0) -> float:
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        numero = minimo
    return max(minimo, min(maximo, numero))


def limpiar_texto(texto: str) -> str:
    return normalizar(str(texto or ""))


def palabras(texto: str) -> List[str]:
    try:
        return [str(x) for x in tokens(texto) if str(x)]
    except Exception:
        return limpiar_texto(texto).split()


def contiene(texto: str, opciones: Iterable[str]) -> bool:
    base = limpiar_texto(texto)
    return any(limpiar_texto(x) in base for x in opciones)


def similitud_tokens(a: str, b: str) -> float:
    sa = set(palabras(a))
    sb = set(palabras(b))
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def estado_vacio() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "tema_actual": None,
        "subtema_actual": None,
        "ultima_intencion": None,
        "ultima_confianza": 0.0,
        "ultima_hipotesis": None,
        "pregunta_pendiente": None,
        "respuesta_esperada": None,
        "ultimo_objeto": None,
        "ultimo_verbo": None,
        "turnos": [],
        "referentes": {},
        "hechos_temporales": {},
        "eventos": [],
        "historial_temas": [],
        "metadatos": {},
    }


class MotorContexto:
    """Mantiene estado y significado conversacional de TAMY."""

    def __init__(self, ruta: os.PathLike[str] | str = DEFAULT_PATH):
        self.ruta = Path(ruta)
        self.estado: Dict[str, Any] = estado_vacio()
        self.cargar()

    # ================================================================
    # PERSISTENCIA
    # ================================================================

    def cargar(self) -> Dict[str, Any]:
        if not self.ruta.exists():
            self.estado = estado_vacio()
            return self.estado
        try:
            with self.ruta.open("r", encoding="utf-8") as archivo:
                datos = json.load(archivo)
            base = estado_vacio()
            if isinstance(datos, dict):
                base.update(datos)
            self.estado = base
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self.estado = estado_vacio()
        return self.estado

    def guardar(self) -> bool:
        try:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            temporal = self.ruta.with_suffix(self.ruta.suffix + ".tmp")
            with temporal.open("w", encoding="utf-8") as archivo:
                json.dump(self.estado, archivo, ensure_ascii=False, indent=4)
            temporal.replace(self.ruta)
            return True
        except OSError:
            return False

    def reiniciar(self) -> None:
        self.estado = estado_vacio()
        self.guardar()

    def snapshot(self) -> Dict[str, Any]:
        return deepcopy(self.estado)

    def restaurar(self, datos: Dict[str, Any]) -> bool:
        if not isinstance(datos, dict):
            return False
        base = estado_vacio()
        base.update(deepcopy(datos))
        self.estado = base
        return self.guardar()

    # ================================================================
    # TURNOS
    # ================================================================

    def registrar_turno(
        self,
        usuario: str,
        respuesta: str = "",
        intencion: Optional[str] = None,
        confianza: float = 0.0,
        tema: Optional[str] = None,
        pregunta_pendiente: Optional[str] = None,
    ) -> Turno:
        turno = Turno(
            id=nuevo_id("turn"),
            usuario=str(usuario),
            respuesta=str(respuesta),
            intencion=intencion,
            confianza=limitar(confianza),
            timestamp=ahora_iso(),
            tema=tema,
            pregunta_pendiente=pregunta_pendiente,
        )
        turnos = self.estado.setdefault("turnos", [])
        turnos.append(turno.to_dict())
        self.estado["turnos"] = turnos[-MAX_HISTORY:]
        self.registrar_evento("turno", usuario)
        self.guardar()
        return turno

    def registrar_respuesta(self, respuesta: str) -> None:
        turnos = self.estado.get("turnos", [])
        if turnos and isinstance(turnos[-1], dict):
            turnos[-1]["respuesta"] = str(respuesta)
            self.estado["turnos"] = turnos[-MAX_HISTORY:]
            self.guardar()

    def ultimos_turnos(self, limite: int = 5) -> List[Dict[str, Any]]:
        try:
            limite = max(1, int(limite))
        except (TypeError, ValueError):
            limite = 5
        return list(self.estado.get("turnos", [])[-limite:])

    def ultimo_turno(self) -> Optional[Dict[str, Any]]:
        turnos = self.estado.get("turnos", [])
        return turnos[-1] if turnos else None

    def turnos_con_tema(self, tema: str) -> List[Dict[str, Any]]:
        buscado = limpiar_texto(tema)
        return [
            turno for turno in self.estado.get("turnos", [])
            if limpiar_texto(turno.get("tema", "")) == buscado
        ]

    # ================================================================
    # TEMA
    # ================================================================

    def establecer_tema(
        self,
        tema: Optional[str],
        subtema: Optional[str] = None,
    ) -> None:
        anterior = self.estado.get("tema_actual")
        self.estado["tema_actual"] = tema
        self.estado["subtema_actual"] = subtema
        if tema and limpiar_texto(tema) != limpiar_texto(anterior or ""):
            historial = self.estado.setdefault("historial_temas", [])
            historial.append({
                "tema": tema,
                "subtema": subtema,
                "timestamp": ahora_iso(),
            })
            self.estado["historial_temas"] = historial[-MAX_TOPICS:]
            self.registrar_evento("tema", str(tema), str(subtema or ""))
        self.guardar()

    def tema_actual(self) -> Optional[str]:
        return self.estado.get("tema_actual")

    def subtema_actual(self) -> Optional[str]:
        return self.estado.get("subtema_actual")

    def tema_es(self, tema: str) -> bool:
        return limpiar_texto(self.tema_actual() or "") == limpiar_texto(tema)

    def cambiar_tema(
        self,
        nuevo_tema: str,
        subtema: Optional[str] = None,
    ) -> bool:
        anterior = self.tema_actual()
        self.establecer_tema(nuevo_tema, subtema)
        self.registrar_evento(
            "cambio_tema",
            f"{anterior or 'ninguno'} -> {nuevo_tema}",
        )
        self.guardar()
        return True

    def historial_temas(self, limite: int = 10) -> List[Dict[str, Any]]:
        historial = self.estado.get("historial_temas", [])
        return list(historial[-max(1, int(limite)):])

    # ================================================================
    # INTENCION E HIPOTESIS
    # ================================================================

    def establecer_intencion(
        self,
        intencion: Optional[str],
        confianza: float = 0.0,
    ) -> None:
        self.estado["ultima_intencion"] = intencion
        self.estado["ultima_confianza"] = limitar(confianza)
        self.estado["ultima_hipotesis"] = intencion
        self.guardar()

    def ultima_intencion(self) -> Optional[str]:
        return self.estado.get("ultima_intencion")

    def ultima_confianza(self) -> float:
        return float(self.estado.get("ultima_confianza", 0.0))

    def hipotesis_actual(self) -> Optional[str]:
        return self.estado.get("ultima_hipotesis")

    # ================================================================
    # PREGUNTAS PENDIENTES
    # ================================================================

    def establecer_pregunta(
        self,
        texto: str,
        tipo_respuesta: str = "libre",
        intencion: Optional[str] = None,
        opciones: Optional[Sequence[str]] = None,
    ) -> PreguntaPendiente:
        pregunta = PreguntaPendiente(
            texto=str(texto),
            tipo_respuesta=str(tipo_respuesta),
            intencion=intencion,
            opciones=list(opciones or []),
            creada_en=ahora_iso(),
            activa=True,
        )
        self.estado["pregunta_pendiente"] = pregunta.to_dict()
        self.estado["respuesta_esperada"] = tipo_respuesta
        self.guardar()
        return pregunta

    def pregunta_pendiente(self) -> Optional[Dict[str, Any]]:
        pregunta = self.estado.get("pregunta_pendiente")
        if isinstance(pregunta, dict) and pregunta.get("activa", True):
            return pregunta
        return None

    def limpiar_pregunta(self) -> None:
        pendiente = self.estado.get("pregunta_pendiente")
        if isinstance(pendiente, dict):
            pendiente["activa"] = False
        self.estado["pregunta_pendiente"] = None
        self.estado["respuesta_esperada"] = None
        self.guardar()

    def esta_esperando(self) -> bool:
        return self.pregunta_pendiente() is not None

    def tipo_respuesta_esperada(self) -> Optional[str]:
        return self.estado.get("respuesta_esperada")

    # ================================================================
    # REFERENTES
    # ================================================================

    def guardar_referente(
        self,
        clave: str,
        valor: str,
        tipo: str = "objeto",
        fuente: str = "conversacion",
        confianza: float = 1.0,
    ) -> Referente:
        referente = Referente(
            clave=limpiar_texto(clave),
            valor=str(valor),
            tipo=str(tipo),
            fuente=str(fuente),
            confianza=limitar(confianza),
            timestamp=ahora_iso(),
        )
        referentes = self.estado.setdefault("referentes", {})
        referentes[referente.clave] = referente.to_dict()
        if len(referentes) > MAX_REFERENTS:
            claves = list(referentes.keys())
            for vieja in claves[:-MAX_REFERENTS]:
                referentes.pop(vieja, None)
        self.estado["referentes"] = referentes
        self.guardar()
        return referente

    def obtener_referente(self, clave: str) -> Optional[str]:
        registro = self.estado.get("referentes", {}).get(limpiar_texto(clave))
        if isinstance(registro, dict):
            return registro.get("valor")
        return None

    def obtener_referente_completo(
        self,
        clave: str,
    ) -> Optional[Dict[str, Any]]:
        registro = self.estado.get("referentes", {}).get(limpiar_texto(clave))
        return registro if isinstance(registro, dict) else None

    def guardar_ultimo_objeto(
        self,
        valor: str,
        tipo: str = "objeto",
        confianza: float = 1.0,
    ) -> None:
        self.estado["ultimo_objeto"] = str(valor)
        self.guardar_referente(
            "ultimo_objeto",
            valor,
            tipo,
            "ultimo_objeto",
            confianza,
        )
        self.guardar_referente(
            "esto",
            valor,
            tipo,
            "referencia",
            confianza,
        )
        self.guardar_referente(
            "eso",
            valor,
            tipo,
            "referencia",
            confianza,
        )
        self.guardar_referente(
            "esta_cosa",
            valor,
            tipo,
            "referencia",
            confianza,
        )

    def guardar_ultimo_verbo(self, verbo: str) -> None:
        self.estado["ultimo_verbo"] = str(verbo)
        self.guardar_referente(
            "ultimo_verbo",
            verbo,
            "verbo",
            "conversacion",
            0.8,
        )

    def ultimo_objeto(self) -> Optional[str]:
        return self.estado.get("ultimo_objeto")

    def ultimo_verbo(self) -> Optional[str]:
        return self.estado.get("ultimo_verbo")

    # ================================================================
    # HECHOS TEMPORALES
    # ================================================================

    def guardar_hecho(
        self,
        clave: str,
        valor: Any,
        fuente: str = "conversacion",
    ) -> None:
        hechos = self.estado.setdefault("hechos_temporales", {})
        hechos[limpiar_texto(clave)] = {
            "valor": valor,
            "fuente": fuente,
            "timestamp": ahora_iso(),
        }
        if len(hechos) > MAX_FACTS:
            claves = list(hechos.keys())
            for vieja in claves[:-MAX_FACTS]:
                hechos.pop(vieja, None)
        self.estado["hechos_temporales"] = hechos
        self.guardar()

    def obtener_hecho(
        self,
        clave: str,
        default: Any = None,
    ) -> Any:
        hecho = self.estado.get("hechos_temporales", {}).get(
            limpiar_texto(clave)
        )
        if isinstance(hecho, dict):
            return hecho.get("valor", default)
        return default

    def hechos(self) -> Dict[str, Any]:
        return dict(self.estado.get("hechos_temporales", {}))

    # ================================================================
    # EVENTOS
    # ================================================================

    def registrar_evento(
        self,
        tipo: str,
        mensaje: str,
        valor: str = "",
    ) -> None:
        eventos = self.estado.setdefault("eventos", [])
        evento = EventoContextual(
            tipo=str(tipo),
            mensaje=str(mensaje),
            valor=str(valor),
            timestamp=ahora_iso(),
        )
        eventos.append(evento.to_dict())
        self.estado["eventos"] = eventos[-MAX_EVENTS:]

    def eventos_recientes(self, limite: int = 10) -> List[Dict[str, Any]]:
        return list(
            self.estado.get("eventos", [])[-max(1, int(limite)):]
        )

    # ================================================================
    # RESPUESTAS CORTAS
    # ================================================================

    @staticmethod
    def es_afirmacion(texto: str) -> bool:
        return limpiar_texto(texto) in {
            "si",
            "s",
            "claro",
            "correcto",
            "exacto",
            "afirmativo",
            "asi es",
            "por supuesto",
            "dale",
            "ok",
            "okay",
            "va",
        }

    @staticmethod
    def es_negacion(texto: str) -> bool:
        return limpiar_texto(texto) in {
            "no",
            "n",
            "nop",
            "negativo",
            "para nada",
        }

    @staticmethod
    def es_confirmacion(texto: str) -> bool:
        base = limpiar_texto(texto)
        return base in {
            "si",
            "s",
            "no",
            "n",
            "claro",
            "correcto",
            "exacto",
            "afirmativo",
            "negativo",
        }

    @staticmethod
    def es_seguimiento(texto: str) -> bool:
        base = limpiar_texto(texto)
        marcadores = {
            "y",
            "tambien",
            "entonces",
            "pero",
            "ademas",
            "ahora",
            "despues",
            "luego",
            "eso",
            "esa",
            "ese",
            "lo mismo",
            "como hago",
            "y como",
            "y eso",
            "y ahora",
            "que sigue",
            "que hago ahora",
        }
        return (
            base in marcadores
            or base.startswith("y ")
            or base.startswith("entonces ")
            or base.startswith("pero ")
        )

    @staticmethod
    def es_cambio_tema(texto: str) -> bool:
        base = limpiar_texto(texto)
        marcadores = {
            "otra cosa",
            "cambiemos de tema",
            "otro tema",
            "por cierto",
            "dejando eso",
            "olvida eso",
            "mejor otra cosa",
        }
        return (
            base in marcadores
            or any(base.startswith(x + " ") for x in marcadores)
        )

    # ================================================================
    # REFERENCIAS Y CORREFERENCIAS
    # ================================================================

    def resolver_referencia(self, texto: str) -> Dict[str, Any]:
        base = limpiar_texto(texto)
        candidatos: List[Dict[str, Any]] = []

        mapa = {
            "esto": "esto",
            "eso": "eso",
            "esa": "eso",
            "ese": "eso",
            "esto mismo": "esto",
            "lo anterior": "ultimo_objeto",
            "eso anterior": "ultimo_objeto",
            "lo mismo": "ultimo_objeto",
            "la fecha": "fecha",
            "el proyecto": "proyecto",
            "la tarea": "tarea",
            "el archivo": "archivo",
            "la imagen": "imagen",
            "esa fecha": "fecha",
            "ese proyecto": "proyecto",
            "esa tarea": "tarea",
        }

        for expresion, clave in mapa.items():
            if expresion in base:
                valor = self.obtener_referente(clave)
                if valor is None:
                    valor = self.obtener_hecho(clave)
                if valor is not None:
                    candidatos.append({
                        "expresion": expresion,
                        "clave": clave,
                        "valor": valor,
                        "confianza": 0.85,
                    })

        if not candidatos:
            valor = self.obtener_referente("ultimo_objeto")
            if valor and base in {
                "eso",
                "esa",
                "ese",
                "lo mismo",
                "lo anterior",
            }:
                candidatos.append({
                    "expresion": base,
                    "clave": "ultimo_objeto",
                    "valor": valor,
                    "confianza": 0.7,
                })

        return {
            "entrada": texto,
            "encontradas": candidatos,
            "principal": candidatos[0] if candidatos else None,
        }

    def hay_referencia(self, texto: str) -> bool:
        return bool(self.resolver_referencia(texto).get("encontradas"))

    # ================================================================
    # INTERPRETACION DE RESPUESTAS
    # ================================================================

    def interpretar_respuesta(self, texto: str) -> Dict[str, Any]:
        base = limpiar_texto(texto)
        pregunta = self.pregunta_pendiente()

        resultado: Dict[str, Any] = {
            "tipo": "libre",
            "valor": texto,
            "confirmacion": None,
            "referencia": self.resolver_referencia(texto).get("principal"),
            "pregunta_pendiente": pregunta,
        }

        if self.es_afirmacion(base):
            resultado["tipo"] = "afirmacion"
            resultado["confirmacion"] = True
            return resultado

        if self.es_negacion(base):
            resultado["tipo"] = "negacion"
            resultado["confirmacion"] = False
            return resultado

        if re.fullmatch(r"\d{1,2}", base):
            resultado["tipo"] = "numero"
            return resultado

        if re.fullmatch(
            r"\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?",
            base,
        ):
            resultado["tipo"] = "fecha"
            return resultado

        if self.es_seguimiento(texto):
            resultado["tipo"] = "seguimiento"
            return resultado

        if self.hay_referencia(texto):
            resultado["tipo"] = "referencia"
            return resultado

        return resultado

    # ================================================================
    # CONTEXTO RESUMIDO
    # ================================================================

    def contexto_resumen(self) -> Dict[str, Any]:
        pregunta = self.pregunta_pendiente()
        return {
            "tema": self.tema_actual(),
            "subtema": self.subtema_actual(),
            "intencion": self.ultima_intencion(),
            "confianza": self.ultima_confianza(),
            "pregunta": pregunta.get("texto") if pregunta else None,
            "respuesta_esperada": self.tipo_respuesta_esperada(),
            "ultimo_objeto": self.ultimo_objeto(),
            "ultimo_verbo": self.ultimo_verbo(),
            "turnos": len(self.estado.get("turnos", [])),
            "referentes": len(self.estado.get("referentes", {})),
        }

    def contexto_para_motor(self) -> Dict[str, Any]:
        return {
            "tema_actual": self.tema_actual(),
            "subtema_actual": self.subtema_actual(),
            "ultima_intencion": self.ultima_intencion(),
            "ultima_confianza": self.ultima_confianza(),
            "pregunta_pendiente": self.pregunta_pendiente(),
            "respuesta_esperada": self.tipo_respuesta_esperada(),
            "ultimo_objeto": self.ultimo_objeto(),
            "ultimo_verbo": self.ultimo_verbo(),
            "referentes": self.estado.get("referentes", {}),
            "hechos": self.estado.get("hechos_temporales", {}),
            "turnos": self.ultimos_turnos(8),
        }

    # ================================================================
    # PUNTUACION CONTEXTUAL
    # ================================================================

    def bono_contextual(
        self,
        mensaje: str,
        intencion: Optional[str],
    ) -> float:
        if not intencion:
            return 0.0

        base = limpiar_texto(mensaje)
        bono = 0.0

        if self.ultima_intencion() == intencion:
            bono += 1.5

        tema = limpiar_texto(self.tema_actual() or "")
        if tema and tema == limpiar_texto(intencion):
            bono += 1.0

        pregunta = self.pregunta_pendiente()
        if pregunta and pregunta.get("intencion") == intencion:
            bono += 5.0

        if self.es_seguimiento(base):
            bono += 1.0

        if self.hay_referencia(base):
            bono += 1.0

        return bono

    def ajustar_resultado(
        self,
        mensaje: str,
        resultado: Dict[str, Any],
    ) -> Dict[str, Any]:
        copia = dict(resultado)
        intencion = copia.get("intencion")
        bono = self.bono_contextual(mensaje, intencion)
        copia["bono_contexto"] = round(bono, 4)
        copia["puntuacion_contextual"] = round(
            float(copia.get("puntuacion", 0.0)) + bono,
            4,
        )
        copia["contexto_aplicado"] = True
        return copia

    def ajustar_resultados(
        self,
        mensaje: str,
        resultados: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ajustados: List[Dict[str, Any]] = []
        for original in resultados:
            copia = self.ajustar_resultado(mensaje, original)
            copia["puntuacion"] = copia["puntuacion_contextual"]
            ajustados.append(copia)
        ajustados.sort(
            key=lambda item: float(item.get("puntuacion", 0.0)),
            reverse=True,
        )
        return ajustados

    # ================================================================
    # INTERPRETACION CONVERSACIONAL
    # ================================================================

    def interpretar_contexto(self, mensaje: str) -> Dict[str, Any]:
        respuesta = self.interpretar_respuesta(mensaje)
        referencia = respuesta.get("referencia")
        pregunta = self.pregunta_pendiente()

        intencion_contextual = None
        if pregunta and pregunta.get("intencion"):
            intencion_contextual = pregunta.get("intencion")
        elif self.ultima_intencion():
            intencion_contextual = self.ultima_intencion()

        tipo = respuesta.get("tipo")

        if tipo == "afirmacion" and pregunta:
            accion = "confirmar_pregunta"
        elif tipo == "negacion" and pregunta:
            accion = "rechazar_pregunta"
        elif tipo in {"referencia", "seguimiento"}:
            accion = "continuar_contexto"
        elif self.es_cambio_tema(mensaje):
            accion = "cambiar_tema"
        else:
            accion = "nuevo_mensaje"

        return {
            "mensaje": mensaje,
            "tipo_respuesta": tipo,
            "accion_contextual": accion,
            "intencion_contextual": intencion_contextual,
            "referencia": referencia,
            "pregunta": pregunta,
            "tema": self.tema_actual(),
            "continuacion": self.es_seguimiento(mensaje),
            "cambio_tema": self.es_cambio_tema(mensaje),
        }

    def analizar_continuacion(
        self,
        mensaje: str,
    ) -> Dict[str, Any]:
        analisis = self.interpretar_contexto(mensaje)
        puntuacion = 0.0

        if analisis["continuacion"]:
            puntuacion += 2.0
        if analisis["referencia"]:
            puntuacion += 3.0
        if analisis["pregunta"]:
            puntuacion += 4.0
        if self.ultima_intencion():
            puntuacion += 1.0

        analisis["puntuacion_continuacion"] = round(puntuacion, 4)
        analisis["es_continuacion"] = puntuacion >= 3.0
        return analisis

    # ================================================================
    # OBSERVACION
    # ================================================================

    def observar_mensaje(
        self,
        mensaje: str,
        intencion: Optional[str] = None,
        confianza: float = 0.0,
        respuesta: str = "",
        tema: Optional[str] = None,
    ) -> Dict[str, Any]:
        analisis = self.interpretar_contexto(mensaje)

        if tema:
            self.establecer_tema(tema)

        if intencion:
            self.establecer_intencion(intencion, confianza)

        if analisis.get("referencia"):
            referencia = analisis["referencia"]
            self.registrar_evento(
                "referencia",
                str(referencia.get("expresion", "")),
                str(referencia.get("valor", "")),
            )

        if analisis.get("accion_contextual") == "cambiar_tema":
            self.registrar_evento(
                "cambio_tema_detectado",
                mensaje,
            )

        if analisis.get("tipo_respuesta") in {
            "afirmacion",
            "negacion",
        }:
            self.registrar_evento(
                "respuesta_corta",
                mensaje,
                analisis["tipo_respuesta"],
            )

        pregunta_activa = self.pregunta_pendiente()
        turno = self.registrar_turno(
            usuario=mensaje,
            respuesta=respuesta,
            intencion=intencion,
            confianza=confianza,
            tema=tema or self.tema_actual(),
            pregunta_pendiente=(
                pregunta_activa.get("texto")
                if pregunta_activa
                else None
            ),
        )

        if respuesta:
            self.registrar_respuesta(respuesta)

        self.estado["metadatos"]["ultimo_analisis"] = analisis
        self.guardar()
        return {
            "analisis": analisis,
            "turno": turno.to_dict(),
        }

    # ================================================================
    # APRENDIZAJE CONTEXTUAL
    # ================================================================

    def aprender_referente(
        self,
        expresion: str,
        valor: str,
        tipo: str = "objeto",
    ) -> None:
        self.guardar_referente(
            expresion,
            valor,
            tipo,
            "aprendizaje_contextual",
            1.0,
        )

    def aprender_tema(
        self,
        tema: str,
        subtema: Optional[str] = None,
    ) -> None:
        self.establecer_tema(tema, subtema)
        self.registrar_evento(
            "tema_aprendido",
            tema,
            subtema or "",
        )
        self.guardar()

    def aprender_respuesta_esperada(self, tipo: str) -> None:
        self.estado["respuesta_esperada"] = tipo
        self.registrar_evento(
            "respuesta_esperada",
            tipo,
        )
        self.guardar()

    def aprender_pregunta(
        self,
        pregunta: str,
        tipo: str,
        intencion: Optional[str] = None,
    ) -> None:
        self.establecer_pregunta(
            pregunta,
            tipo,
            intencion,
        )
        self.registrar_evento(
            "pregunta_aprendida",
            pregunta,
            tipo,
        )
        self.guardar()

    # ================================================================
    # ATAJOS DE CONTEXTO PARA TAMY
    # ================================================================

    def contexto_de_si(self) -> Dict[str, Any]:
        pregunta = self.pregunta_pendiente()
        return {
            "respuesta": "si",
            "significa": "confirmacion",
            "pregunta": pregunta,
            "intencion": pregunta.get("intencion") if pregunta else self.ultima_intencion(),
        }

    def contexto_de_no(self) -> Dict[str, Any]:
        pregunta = self.pregunta_pendiente()
        return {
            "respuesta": "no",
            "significa": "rechazo",
            "pregunta": pregunta,
            "intencion": pregunta.get("intencion") if pregunta else self.ultima_intencion(),
        }

    def contexto_de_eso(self) -> Dict[str, Any]:
        referencia = self.resolver_referencia("eso")
        return {
            "referencia": referencia,
            "valor": referencia.get("principal", {}).get("valor")
            if referencia.get("principal")
            else None,
        }

    def contexto_de_esa(self) -> Dict[str, Any]:
        referencia = self.resolver_referencia("esa")
        return {
            "referencia": referencia,
            "valor": referencia.get("principal", {}).get("valor")
            if referencia.get("principal")
            else None,
        }

    def contexto_de_ese(self) -> Dict[str, Any]:
        referencia = self.resolver_referencia("ese")
        return {
            "referencia": referencia,
            "valor": referencia.get("principal", {}).get("valor")
            if referencia.get("principal")
            else None,
        }

    # ================================================================
    # DIAGNOSTICO
    # ================================================================

    def diagnostico(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "tema": self.tema_actual(),
            "subtema": self.subtema_actual(),
            "intencion": self.ultima_intencion(),
            "confianza": self.ultima_confianza(),
            "esperando_respuesta": self.esta_esperando(),
            "tipo_respuesta": self.tipo_respuesta_esperada(),
            "ultimo_objeto": self.ultimo_objeto(),
            "ultimo_verbo": self.ultimo_verbo(),
            "turnos": len(self.estado.get("turnos", [])),
            "referentes": len(self.estado.get("referentes", {})),
            "hechos": len(self.estado.get("hechos_temporales", {})),
            "eventos": len(self.estado.get("eventos", [])),
            "temas": len(self.estado.get("historial_temas", [])),
        }

    def resumen_texto(self) -> str:
        datos = self.diagnostico()
        return (
            f"TEMA={datos['tema']!r} | "
            f"SUBTEMA={datos['subtema']!r} | "
            f"INTENCION={datos['intencion']!r} | "
            f"CONFIANZA={datos['confianza']:.3f} | "
            f"ESPERANDO={datos['esperando_respuesta']} | "
            f"OBJETO={datos['ultimo_objeto']!r} | "
            f"TURNOS={datos['turnos']} | "
            f"REFERENTES={datos['referentes']}"
        )

    # ================================================================
    # EXPORTACION
    # ================================================================

    def exportar(
        self,
        ruta: os.PathLike[str] | str,
    ) -> bool:
        try:
            destino = Path(ruta)
            destino.parent.mkdir(parents=True, exist_ok=True)
            with destino.open("w", encoding="utf-8") as archivo:
                json.dump(
                    self.estado,
                    archivo,
                    ensure_ascii=False,
                    indent=4,
                )
            return True
        except OSError:
            return False

    def importar(
        self,
        ruta: os.PathLike[str] | str,
        reemplazar: bool = True,
    ) -> bool:
        try:
            with Path(ruta).open("r", encoding="utf-8") as archivo:
                datos = json.load(archivo)
        except (OSError, json.JSONDecodeError):
            return False

        if not isinstance(datos, dict):
            return False

        if reemplazar:
            return self.restaurar(datos)

        combinado = self.snapshot()
        combinado.update(datos)
        return self.restaurar(combinado)


_motor_global: Optional[MotorContexto] = None


def crear_motor(
    ruta: os.PathLike[str] | str = DEFAULT_PATH,
) -> MotorContexto:
    return MotorContexto(ruta)


def obtener_motor(
    ruta: os.PathLike[str] | str = DEFAULT_PATH,
) -> MotorContexto:
    global _motor_global
    if _motor_global is None:
        _motor_global = MotorContexto(ruta)
    return _motor_global


def contexto_actual() -> Dict[str, Any]:
    return obtener_motor().contexto_para_motor()


def interpretar(mensaje: str) -> Dict[str, Any]:
    return obtener_motor().interpretar_contexto(mensaje)


def analizar_continuacion(mensaje: str) -> Dict[str, Any]:
    return obtener_motor().analizar_continuacion(mensaje)


def resolver_referencia(mensaje: str) -> Dict[str, Any]:
    return obtener_motor().resolver_referencia(mensaje)


def es_afirmacion(mensaje: str) -> bool:
    return MotorContexto.es_afirmacion(mensaje)


def es_negacion(mensaje: str) -> bool:
    return MotorContexto.es_negacion(mensaje)


def diagnostico() -> Dict[str, Any]:
    return obtener_motor().diagnostico()


def ejecutar_pruebas() -> None:
    ruta = Path("data/tamy_context_test.json")
    motor = MotorContexto(ruta)
    motor.reiniciar()
    motor.establecer_tema("proyecto", "TAMY")
    motor.establecer_intencion("trabajar", 0.8)
    motor.establecer_pregunta(
        "¿Quieres que abra el proyecto?",
        "si_no",
        "abrir_proyecto",
    )
    motor.guardar_ultimo_objeto(
        "proyecto TAMY",
        "proyecto",
    )

    assert motor.es_afirmacion("si")
    assert motor.es_negacion("no")
    assert motor.esta_esperando()

    analisis = motor.interpretar_contexto("eso")
    assert analisis["accion_contextual"] == "continuar_contexto"
    assert analisis["referencia"] is not None

    print("MOTOR DE CONTEXTO: OK")
    print(motor.resumen_texto())

    try:
        ruta.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    ejecutar_pruebas()