#!/usr/bin/env python3
"""
Casos de uso del sistema como subrutinas de GStreamer (gi), en el mismo
estilo de gstreamer.py. Se ejecuta en el host machine y es independiente de
los demás archivos de esta carpeta: solo necesita los dos modelos .onnx.

Caso 1: Configuración y mantenimiento (Ingeniero)
    caso1_ver_camara()            muestra la cámara en pantalla
    caso1_medir_fps()             mide los FPS reales de la cámara
    caso1_enrolar(nombre)         registra un usuario (cámara o foto)
    caso1_listar_usuarios()       lista los usuarios registrados
    caso1_eliminar_usuario(nombre)
Caso 2: Observación y supervisión (Vigilante)
    caso2_ver_en_vivo()           video en vivo recibido por UDP/RTP
    caso2_revisar_evidencia()     navega la evidencia guardada en disco
Caso 3: Accionamiento (usuario registrado o no registrado)
    caso3_accionamiento()         transmite, reconoce y decide
Caso 4: Abrir
    caso4_abrir_puerta()          activa la salida eléctrica (simulada en PC)

El pipeline del caso 3 se parte con `tee` en dos ramas, cada una seguida de
`queue`: una rama entrega cuadros BGR a Python (appsink) y la otra transmite
H264/RTP por UDP hacia el vigilante.
"""
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

import re
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# Inicializar GStreamer
Gst.init(None)

# ------------------------------------------------------------ configuración --
DISPOSITIVO = "/dev/video0"
ANCHO, ALTO = 1280, 720
IP_DESTINO = "127.0.0.1"          # IP de la PC del vigilante
PUERTO_UDP = 5000
UMBRAL = 0.363                    # similitud coseno mínima (SFace)
SEGUNDOS_ENTRE_ANALISIS = 0.2     # ~5 análisis por segundo
SEGUNDOS_ENTRE_APERTURAS = 5
SEGUNDOS_ENTRE_EVIDENCIAS = 2
PIN_LED = 17                      # solo se usa si hay GPIO (Raspberry Pi)
FPS_REQUERIDOS = 60

# Cámara MJPG: es el modo que entrega 1280x720 en esta webcam. El framerate no
# se fija porque la cámara lo anuncia como 2997/100 y un filtro "30/1" falla.
FUENTE_CAMARA = (f"v4l2src device={DISPOSITIVO} ! "
                 f"image/jpeg,width={ANCHO},height={ALTO} ! jpegdec")
# Para una cámara que entregue video crudo, usa en su lugar:
# FUENTE_CAMARA = (f"v4l2src device={DISPOSITIVO} ! "
#                  f"video/x-raw,width={ANCHO},height={ALTO}")

DIR = Path(__file__).resolve().parent
DIR_USUARIOS = DIR / "usuarios"
DIR_EVIDENCIA = DIR / "evidencia"
MODELO_DETECTOR = DIR / "face_detection_yunet_2023mar.onnx"
MODELO_RECONOCEDOR = DIR / "face_recognition_sface_2021dec.onnx"


# ------------------------------------------------------- pipelines de apoyo --
def _pipeline_captura_bgr() -> str:
    """Cámara -> cuadros BGR para Python (sin transmisión)."""
    return (f"{FUENTE_CAMARA} ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true drop=true max-buffers=1 sync=false")


def _pipeline_transmision() -> str:
    """Pipeline principal del caso 3: tee con dos ramas, cada una con queue."""
    return (
        f"{FUENTE_CAMARA} ! videoconvert ! tee name=t "
        # Rama 1: Python (la cola es 'leaky' para no frenar la transmisión)
        "t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=true drop=true max-buffers=1 sync=false "
        # Rama 2: red
        "t. ! queue ! videoconvert ! "
        "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 ! "
        f"rtph264pay pt=96 config-interval=1 ! udpsink host={IP_DESTINO} port={PUERTO_UDP}"
    )


def _pipeline_recepcion() -> str:
    """Recepción RTP/H264 por UDP y despliegue en pantalla."""
    return (
        f"udpsrc port={PUERTO_UDP} "
        'caps="application/x-rtp,media=video,clock-rate=90000,'
        'encoding-name=H264,payload=96" ! '
        "rtpjitterbuffer latency=50 ! rtph264depay ! h264parse ! "
        "avdec_h264 ! videoconvert ! autovideosink sync=false"
    )


# ------------------------------------------------------ ejecución de pipelines --
def _al_mensaje(_bus, msg, loop):
    if msg.type == Gst.MessageType.ERROR:
        err, depuracion = msg.parse_error()
        print(f"[GSTREAMER] Error: {err.message}\n{depuracion}")
        loop.quit()
    elif msg.type == Gst.MessageType.EOS:
        loop.quit()


def _terminar(loop):
    loop.quit()
    return False  # no repetir el temporizador


def _muestra_a_numpy(muestra):
    """Convierte un sample BGR de GStreamer en un arreglo (alto, ancho, 3)."""
    buf = muestra.get_buffer()
    estructura = muestra.get_caps().get_structure(0)
    ancho, alto = estructura.get_value("width"), estructura.get_value("height")
    ok, info = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        datos = np.frombuffer(info.data, dtype=np.uint8)
        paso = len(datos) // alto  # bytes por fila (puede incluir relleno)
        frame = datos.reshape(alto, paso)[:, :ancho * 3].reshape(alto, ancho, 3)
        return frame.copy()
    finally:
        buf.unmap(info)


def _al_nuevo_cuadro(sink, al_cuadro):
    muestra = sink.emit("pull-sample")
    if muestra is not None:
        frame = _muestra_a_numpy(muestra)
        if frame is not None:
            al_cuadro(frame)
    return Gst.FlowReturn.OK


def _ejecutar_pipeline(pipeline_str: str, al_cuadro=None, duracion=None):
    """Subrutina auxiliar para construir y reproducir cualquier pipeline.

    - al_cuadro: función que recibe cada cuadro BGR (el pipeline debe tener un
      appsink llamado 'sink' con emit-signals=true). Debe ser rápida.
    - duracion: segundos tras los cuales se detiene solo (None = hasta Ctrl+C).
    """
    pipeline = Gst.parse_launch(pipeline_str)
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", _al_mensaje, loop)

    if al_cuadro is not None:
        sink = pipeline.get_by_name("sink")
        sink.connect("new-sample", _al_nuevo_cuadro, al_cuadro)

    if duracion is not None:
        GLib.timeout_add(int(duracion * 1000), _terminar, loop)

    pipeline.set_state(Gst.State.PLAYING)
    print("Ejecutando pipeline. Presiona Ctrl+C para detener...")
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nDeteniendo pipeline...")
    finally:
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()


# ------------------------------------------------- modelos, usuarios, evidencia --
def _cargar_modelos():
    for modelo in (MODELO_DETECTOR, MODELO_RECONOCEDOR):
        if not modelo.exists():
            raise FileNotFoundError(f"Falta el modelo {modelo.name} en {DIR}")
    detector = cv2.FaceDetectorYN.create(str(MODELO_DETECTOR), "", (320, 320))
    reconocedor = cv2.FaceRecognizerSF.create(str(MODELO_RECONOCEDOR), "")
    return detector, reconocedor


def _detectar_caras(detector, img):
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, caras = detector.detect(img)
    return [] if caras is None else list(caras)


def _embedding_de(reconocedor, img, cara):
    return reconocedor.feature(reconocedor.alignCrop(img, cara))


def _cargar_usuarios():
    """Devuelve {nombre: embedding} de los usuarios registrados."""
    if not DIR_USUARIOS.exists():
        return {}
    return {f.stem: np.load(f) for f in sorted(DIR_USUARIOS.glob("*.npy"))}


def _identificar(reconocedor, embedding, usuarios):
    """Compara contra todos los registrados. Devuelve (nombre|None, score)."""
    mejor_nombre, mejor_score = None, -1.0
    for nombre, ref in usuarios.items():
        score = reconocedor.match(ref, embedding, cv2.FaceRecognizerSF_FR_COSINE)
        if score > mejor_score:
            mejor_nombre, mejor_score = nombre, score
    if mejor_score >= UMBRAL:
        return mejor_nombre, mejor_score
    return None, mejor_score


def _guardar_evidencia(frame, resultado, detalle=""):
    """Guarda el cuadro en disco y lo anota en evidencia/registro.csv."""
    DIR_EVIDENCIA.mkdir(exist_ok=True)
    ahora = time.time()
    marca = time.strftime("%Y%m%d_%H%M%S", time.localtime(ahora))
    nombre = f"{marca}_{int(ahora * 1000) % 1000:03d}_{resultado}.jpg"
    cv2.imwrite(str(DIR_EVIDENCIA / nombre), frame)
    fecha = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ahora))
    with open(DIR_EVIDENCIA / "registro.csv", "a", encoding="utf-8") as f:
        f.write(f"{fecha},{resultado},{detalle},{nombre}\n")
    return nombre


# ======================================================================
# CASO 1: Configuración y mantenimiento (Ingeniero)
# ======================================================================
def caso1_ver_camara():
    """Muestra la cámara en pantalla (equivale a reproducir_webcam)."""
    _ejecutar_pipeline(f"{FUENTE_CAMARA} ! videoconvert ! autovideosink")


def caso1_medir_fps(segundos: float = 5):
    """Mide los FPS reales que entrega la cámara con la configuración actual."""
    t = {"primero": None, "ultimo": None, "n": 0}

    def contar(_frame):
        ahora = time.time()
        if t["primero"] is None:
            t["primero"] = ahora
        t["ultimo"] = ahora
        t["n"] += 1

    _ejecutar_pipeline(_pipeline_captura_bgr(), al_cuadro=contar, duracion=segundos)

    if t["n"] < 2 or t["ultimo"] == t["primero"]:
        print("[CAMARA] No se recibieron suficientes cuadros para medir.")
        return None
    fps = (t["n"] - 1) / (t["ultimo"] - t["primero"])
    print(f"[CAMARA] FPS medidos: {fps:.1f} (requisito: {FPS_REQUERIDOS})")
    if fps < 0.9 * FPS_REQUERIDOS:
        print("[CAMARA] Por debajo del requisito; documenta el valor medido.")
    return fps


def caso1_enrolar(nombre: str, imagen: str = None):
    """Registra un usuario con la cámara o con una foto existente."""
    if not re.fullmatch(r"[\w\-]+", nombre):
        print("[ERROR] El nombre solo puede tener letras, números, _ y -.")
        return
    detector, reconocedor = _cargar_modelos()

    if imagen:
        img = cv2.imread(imagen)
        if img is None:
            print(f"[ERROR] No se pudo leer la imagen {imagen}")
            return
    else:
        print("[ENROLAR] Mira a la cámara. Capturando durante 3 segundos...")
        ultimo = {}
        _ejecutar_pipeline(_pipeline_captura_bgr(),
                           al_cuadro=lambda f: ultimo.__setitem__("frame", f),
                           duracion=3)
        img = ultimo.get("frame")  # el último cuadro, ya con la exposición ajustada
        if img is None:
            print("[ERROR] No se pudo tomar la foto.")
            return

    caras = _detectar_caras(detector, img)
    if not caras:
        print("[ERROR] No se detectó ninguna cara. Intenta de nuevo.")
        return
    if len(caras) > 1:
        print("[AVISO] Se detectaron varias caras; se usará la más grande.")
    cara = max(caras, key=lambda f: f[2] * f[3])

    DIR_USUARIOS.mkdir(exist_ok=True)
    np.save(DIR_USUARIOS / f"{nombre}.npy", _embedding_de(reconocedor, img, cara))
    cv2.imwrite(str(DIR_USUARIOS / f"{nombre}.jpg"), img)
    print(f"[ENROLAR] Usuario '{nombre}' registrado.")


def caso1_listar_usuarios():
    usuarios = _cargar_usuarios()
    if not usuarios:
        print("[USUARIOS] No hay usuarios registrados.")
        return
    print(f"[USUARIOS] {len(usuarios)} registrado(s):")
    for nombre in usuarios:
        print(f"  - {nombre}")


def caso1_eliminar_usuario(nombre: str):
    borrados = 0
    for ext in (".npy", ".jpg"):
        archivo = DIR_USUARIOS / f"{nombre}{ext}"
        if archivo.exists():
            archivo.unlink()
            borrados += 1
    if borrados:
        print(f"[USUARIOS] Usuario '{nombre}' eliminado.")
    else:
        print(f"[USUARIOS] No existe el usuario '{nombre}'.")


# ======================================================================
# CASO 2: Observación y supervisión (Vigilante)
# ======================================================================
def caso2_ver_en_vivo():
    """Muestra la transmisión en vivo que envía el caso 3 (rama de red)."""
    print(f"[VIGILANTE] Esperando video en el puerto UDP {PUERTO_UDP}...")
    _ejecutar_pipeline(_pipeline_recepcion())


def caso2_revisar_evidencia():
    """Navega la evidencia guardada: a/d = anterior/siguiente, q = salir."""
    archivos = sorted(DIR_EVIDENCIA.glob("*.jpg")) if DIR_EVIDENCIA.exists() else []
    if not archivos:
        print("[VIGILANTE] Aún no hay evidencia registrada.")
        return
    registro = DIR_EVIDENCIA / "registro.csv"
    if registro.exists():
        print("fecha,resultado,detalle,archivo")
        print(registro.read_text(encoding="utf-8"), end="")

    i = len(archivos) - 1  # empieza por la más reciente
    while True:
        img = cv2.imread(str(archivos[i]))
        if img is not None:
            cv2.putText(img, f"[{i + 1}/{len(archivos)}] {archivos[i].name}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Evidencia (a/d = anterior/siguiente, q = salir)", img)
        tecla = cv2.waitKey(0) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla in (ord("d"), ord("n")):
            i = min(i + 1, len(archivos) - 1)
        elif tecla in (ord("a"), ord("p")):
            i = max(i - 1, 0)
    cv2.destroyAllWindows()


# ======================================================================
# CASO 4: Abrir (extiende el caso 3)
# ======================================================================
_led = None
_led_probado = False


def _obtener_led():
    """Crea el LED una sola vez; None si no hay GPIO (se simula)."""
    global _led, _led_probado
    if not _led_probado:
        _led_probado = True
        try:
            from gpiozero import LED
            _led = LED(PIN_LED)
        except Exception:
            _led = None
    return _led


def caso4_abrir_puerta(segundos: float = 3.0, motivo: str = ""):
    """Activa la salida eléctrica durante `segundos` sin bloquear el programa."""
    led = _obtener_led()
    destino = "LED" if led is not None else "simulación"
    print(f"[PUERTA] ABIERTA ({destino}) durante {segundos:.0f} s. {motivo}")
    if led is not None:
        led.on()

    def cerrar():
        if led is not None:
            led.off()
        print("[PUERTA] Cerrada.")

    threading.Timer(segundos, cerrar).start()


# ======================================================================
# CASO 3: Accionamiento (usuario registrado o no registrado)
# ======================================================================
def caso3_accionamiento():
    """Transmite el video y reconoce rostros contra los usuarios registrados.

    - Registrado: caso 4 (abrir la puerta) y evidencia en disco.
    - No registrado: la puerta permanece cerrada y se guarda evidencia.
    """
    detector, reconocedor = _cargar_modelos()
    usuarios = _cargar_usuarios()
    if usuarios:
        print(f"[SISTEMA] Usuarios registrados: {', '.join(usuarios)}")
    else:
        print("[AVISO] No hay usuarios registrados (usa caso1_enrolar). "
              "Todas las personas se tratarán como no registradas.")

    # El callback de GStreamer solo guarda el último cuadro (es rápido); el
    # reconocimiento corre en un hilo aparte para no frenar el pipeline.
    estado = {"frame": None, "id": 0}
    candado = threading.Lock()
    corriendo = threading.Event()
    corriendo.set()

    def al_cuadro(frame):
        with candado:
            estado["frame"] = frame
            estado["id"] += 1

    def analizar():
        ultimo_id = 0
        ultima_apertura = 0.0
        ultima_evidencia = 0.0
        while corriendo.is_set():
            with candado:
                frame, fid = estado["frame"], estado["id"]
            if frame is None or fid == ultimo_id:
                time.sleep(0.01)
                continue
            ultimo_id = fid
            try:
                ahora = time.time()
                for cara in _detectar_caras(detector, frame):
                    emb = _embedding_de(reconocedor, frame, cara)
                    nombre, score = _identificar(reconocedor, emb, usuarios)
                    if nombre is not None:
                        if ahora - ultima_apertura >= SEGUNDOS_ENTRE_APERTURAS:
                            print(f"[RECONOCIDO] {nombre} (score {score:.2f})")
                            caso4_abrir_puerta(motivo=f"Usuario: {nombre}")
                            _guardar_evidencia(frame, "registrado",
                                               f"{nombre};{score:.2f}")
                            ultima_apertura = ahora
                    elif ahora - ultima_evidencia >= SEGUNDOS_ENTRE_EVIDENCIAS:
                        print(f"[DENEGADO] Persona no registrada (score {score:.2f})")
                        _guardar_evidencia(frame, "desconocido", f"{score:.2f}")
                        ultima_evidencia = ahora
            except Exception:
                traceback.print_exc()
            time.sleep(SEGUNDOS_ENTRE_ANALISIS)

    hilo = threading.Thread(target=analizar, daemon=True)
    hilo.start()
    print(f"[SISTEMA] Transmitiendo a {IP_DESTINO}:{PUERTO_UDP} y monitoreando acceso.")
    try:
        _ejecutar_pipeline(_pipeline_transmision(), al_cuadro=al_cuadro)
    finally:
        corriendo.clear()
        hilo.join(timeout=2)


# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Descomenta la subrutina que quieras probar:

    caso1_ver_camara()
    # caso1_medir_fps()
    # caso1_enrolar("sherman")
    # caso1_listar_usuarios()
    # caso1_eliminar_usuario("sherman")
    # caso2_ver_en_vivo()
    # caso2_revisar_evidencia()
    # caso3_accionamiento()
    # caso4_abrir_puerta()
