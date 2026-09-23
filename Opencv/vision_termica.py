
import cv2
import numpy as np


def main():
    # 1. Inicializar la cámara web de la laptop.
    # El índice '/dev/video0' indica el dispositivo de cámara
    # que verificamos previamente que funciona correctamente.
    cap = cv2.VideoCapture('/dev/video0')

    # Verificar si la cámara se abrió correctamente.
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    print("--- Visión Térmica Iniciada ---")
    print("Presiona 'q' o ESC para cerrar la aplicación.")
    print("Presiona '1' para usar el mapa JET.")
    print("Presiona '2' para usar el mapa INFERNO.")
    print("Presiona '3' para usar el mapa TURBO.")

    # Variable que almacena el mapa de colores que utilizaremos.
    # Inicialmente utilizamos JET.
    current_colormap = cv2.COLORMAP_JET

    try:
        while True:
            # 2. Adquisición: Capturar el fotograma (frame) actual de la cámara.
            # 'ret' indica si la captura fue exitosa y 'frame' contiene
            # la imagen obtenida de la cámara.
            ret, frame = cap.read()

            if not ret:
                print("Error al recibir el stream de video.")
                break

            # 3. Preprocesamiento: Convertir el fotograma de color
            # (BGR) a escala de grises.
            # En lugar de trabajar con tres canales de color (B, G y R),
            # ahora trabajamos con una matriz 2D de intensidades.
            # Esto simplifica las operaciones posteriores.
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # 4. Suavizado: Aplicar un filtro Gaussiano.
            # El desenfoque reduce pequeñas variaciones y ruido en la imagen.
            # Esto ayuda a obtener una imagen más estable para los siguientes
            # pasos de procesamiento.
            blurred = cv2.GaussianBlur(
                gray,
                (7, 7),
                0
            )

            # 5. Visualización mediante un mapa de colores.
            # La imagen en escala de grises contiene diferentes niveles
            # de intensidad. applyColorMap transforma esos niveles
            # de intensidad en colores.
            #
            # Importante: esto NO convierte la cámara en una cámara térmica.
            # Es solamente una representación visual de las intensidades
            # de la imagen utilizando colores.
            thermal = cv2.applyColorMap(
                blurred,
                current_colormap
            )

            # 6. Extracción de características: Detectar bordes
            # utilizando el Operador Canny.
            #
            # Canny busca cambios bruscos de intensidad entre píxeles.
            # Estos cambios suelen corresponder a los límites de objetos
            # presentes en la imagen.
            edges = cv2.Canny(
                blurred,
                50,
                150
            )

            # La salida de Canny es una imagen en escala de grises.
            # Convertimos esa imagen nuevamente a BGR para poder
            # combinarla posteriormente con la imagen coloreada.
            edges_color = cv2.cvtColor(
                edges,
                cv2.COLOR_GRAY2BGR
            )

            # 7. Composición: Combinar la imagen coloreada con los bordes.
            #
            # addWeighted permite mezclar dos imágenes.
            # La primera imagen aporta principalmente los colores,
            # mientras que la segunda aporta los bordes detectados.
            #
            # El resultado es un efecto visual parecido a una
            # "visión de energía".
            result = cv2.addWeighted(
                thermal,
                0.85,
                edges_color,
                0.5,
                0
            )

            # 8. Anotación: Agregar información sobre la imagen.
            # putText permite dibujar texto directamente sobre el frame.
            # Esto es útil para mostrar información al usuario
            # sin necesidad de crear otra interfaz.
            cv2.putText(
                result,
                "VISION TERMICA",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Mostrar las teclas disponibles directamente en la ventana.
            cv2.putText(
                result,
                "1:JET  2:INFERNO  3:TURBO",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # 9. Visualización: Mostrar el resultado final.
            # imshow crea una ventana y muestra el frame procesado.
            # Como estamos dentro de un bucle, la imagen se actualiza
            # continuamente y obtenemos video en tiempo real.
            cv2.imshow(
                'Vision Termica - OpenCV',
                result
            )

            # 10. Control del teclado.
            # waitKey permite que OpenCV procese eventos de la ventana
            # y nos permite detectar teclas presionadas.
            #
            # 'q' o ESC → salir del programa.
            # '1' → seleccionar el mapa JET.
            # '2' → seleccionar el mapa INFERNO.
            # '3' → seleccionar el mapa TURBO.
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break

            elif key == ord('1'):
                current_colormap = cv2.COLORMAP_JET

            elif key == ord('2'):
                current_colormap = cv2.COLORMAP_INFERNO

            elif key == ord('3'):
                current_colormap = cv2.COLORMAP_TURBO

    except KeyboardInterrupt:
        # Permitir cerrar el programa utilizando Ctrl+C
        # desde la terminal.
        print("\nPrograma detenido con Ctrl+C.")

    finally:
        # 11. Liberar recursos.
        # release() libera la cámara para que otros programas
        # puedan utilizarla nuevamente.
        cap.release()

        # destroyAllWindows() cierra todas las ventanas creadas
        # por OpenCV.
        cv2.destroyAllWindows()

        # Procesar los últimos eventos de la interfaz gráfica
        # para asegurarnos de que las ventanas se cierren correctamente.
        cv2.waitKey(1)

        print("Cámara liberada. Programa terminado.")


# Esta condición garantiza que main() se ejecute solamente
# cuando este archivo se ejecuta directamente.
if __name__ == '__main__':
    main()
