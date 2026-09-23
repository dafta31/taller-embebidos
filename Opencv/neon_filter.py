
import cv2
import numpy as np


def main():
    # 1. Inicializar la cámara web de la laptop.
    cap = cv2.VideoCapture('/dev/video0')

    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    print("--- Filtro de Neón Iniciado ---")
    print("Presiona 'q' o ESC para cerrar la aplicación.")

    try:
        while True:
            # 2. Adquisición: capturar el fotograma actual
            ret, frame = cap.read()

            if not ret:
                print("Error al recibir el stream de video.")
                break

            # 3. Preprocesamiento: Convertir el fotograma de color (BGR) a escala de grises
            # Esto reduce el canal de color 3D a una matriz plana de 2D, acelerando el proceso
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 4. Suavizado: Aplicar Gaussian Blur para eliminar el "ruido visual"
            # Usamos un kernel de 5x5. Esto hace que los bordes finales se vean más limpios
            blurred = cv2.GaussianBlur(
                gray,
                (5, 5),
                1.5
            )

            # 5. Extracción de Características: Detectar contornos usando el Operador Canny
            # Analiza los gradientes de intensidad y detecta dónde cambian bruscamente los píxeles
            # Parámetros: imagen suavizada, umbral mínimo (30) y umbral máximo (90)
            edges = cv2.Canny(
                blurred,
                30,
                90
            )

            # 6. Efecto estético "Neón" (Opcional):
            # El resultado de Canny es blanco y negro. Podemos darle color de neón (ej. verde)
            # Creamos una matriz vacía del mismo tamaño que la original en color
            neon_frame = np.zeros_like(frame)

            # Asignamos el canal verde (índice 1 en BGR) donde se detectaron bordes
            neon_frame[edges > 0] = [0, 255, 0]

            # 7. Visualización: Mostrar las ventanas con los resultados en tiempo real   
            cv2.imshow('Filtro Neon en Tiempo Real (Canny)', neon_frame)

            # 8. Control del teclado
            # q = salir
            # ESC = salir
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        # Permite cerrar el programa con Ctrl+C desde la terminal
        print("\nPrograma detenido con Ctrl+C.")

    finally:
        # 9. Liberar recursos siempre, incluso si ocurre un error
        cap.release()
        cv2.destroyAllWindows()

        # Dar tiempo a OpenCV para cerrar las ventanas
        cv2.waitKey(1)

        print("Cámara liberada. Programa terminado.")


if __name__ == '__main__':
    main()
