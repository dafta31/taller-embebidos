import os
import sys
import time
import cv2

# IP de la PC del oficial de seguridad (En pruebas locales puedes usar 127.0.0.1)
IP_PC_GUARDIA = "127.0.0.1"
PUERTO_UDP = 5000

# Pipeline comprobado en la terminal adaptado con appsink para OpenCV
GSTREAMER_PIPELINE = (
    "v4l2src device=/dev/video0 ! videoconvert ! tee name=t "
    "t. ! queue ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1 "
    f"t. ! queue ! x264enc tune=zerolatency speed-preset=ultrafast ! rtph264pay pt=96 ! udpsink host={IP_PC_GUARDIA} port={PUERTO_UDP}"
)

def abrir_puerta():
    """Subrutina para la activación de la cerradura/puerta."""
    print("[CERRADURA] ¡Acceso concedido! Abriendo puerta...")

# -------------------------------------------------------------------------
# PASO 1: Tomar foto de referencia si no existe
# -------------------------------------------------------------------------
if not os.path.exists("referencia.jpg"):
    print("[FOTO] No se encontró 'referencia.jpg'. Sonríe a la cámara...")
    cap_temp = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap_temp.isOpened():
        sys.exit("Error: No se pudo abrir el pipeline para la foto de referencia.")
    
    time.sleep(2)  # Dar tiempo a la cámara para ajustar brillo/exposición
    ok, frame = cap_temp.read()
    if ok and frame is not None:
        cv2.imwrite("referencia.jpg", frame)
        print("[FOTO] ¡Foto de referencia guardada!")
    else:
        sys.exit("[FOTO] Error al tomar la foto de referencia.")
    cap_temp.release()

# -------------------------------------------------------------------------
# PASO 2: Cargar modelos YuNet y SFace
# -------------------------------------------------------------------------
detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")
UMBRAL = 0.363

def obtener_embedding(img):
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, caras = detector.detect(img)
    if caras is None or len(caras) == 0:
        return None
    cara_alineada = recognizer.alignCrop(img, caras[0])
    return recognizer.feature(cara_alineada)

ref_img = cv2.imread("referencia.jpg")
ref_embedding = obtener_embedding(ref_img)

if ref_embedding is None:
    sys.exit("[ERROR] No se detectó ninguna cara en 'referencia.jpg'. Borra la imagen y vuelve a intentar.")

print("[SISTEMA] Referencia enrolada correctamente.")

# -------------------------------------------------------------------------
# PASO 3: Ejecución en vivo (Reconocimiento Local + Transmisión por Red)
# -------------------------------------------------------------------------
cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    sys.exit("[ERROR] No se pudo abrir el pipeline de GStreamer en el emisor.")

print("[SISTEMA] Transmitiendo video y monitoreando acceso...")

while True:
    ok, frame = cap.read()
    if not ok:
        print("[SISTEMA] Error al leer fotograma de GStreamer.")
        break

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, caras = detector.detect(frame)

    if caras is not None:
        for c in caras:
            cara_alineada = recognizer.alignCrop(frame, c)
            emb = recognizer.feature(cara_alineada)
            score = recognizer.match(ref_embedding, emb, cv2.FaceRecognizerSF_FR_COSINE)
            
            if score >= UMBRAL:
                print(f"[RECONOCIDO] Usuario autenticado (Score: {score:.2f})")
                abrir_puerta()

cap.release()