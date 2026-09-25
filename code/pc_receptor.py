import cv2
import socket
import numpy as np
import queue
import threading
import time

# Configuración del servidor
HOST = '0.0.0.0'  # Escucha en todas las interfaces de red
PORT = 5000       # Debe coincidir con el puerto del emisor

# Cola para almacenar solo la imagen más reciente (maxsize=1 evita acumulación)
frame_queue = queue.Queue(maxsize=1)
running = True

def recibir_datos():
    """ Hilo secundario: Recibe paquetes de red sin congelar la ventana gráfica """
    global running
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Permite reutilizar el puerto inmediatamente si se reinicia
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"[RECEPTOR] Escuchando en el puerto {PORT}...")
        
        conn, addr = server_socket.accept()
        print(f"[RECEPTOR] Conectado con Raspberry Pi en {addr}")
        
        data_buffer = b""
        
        while running:
            # Recibir fragmentos de datos
            packet = conn.recv(4096)
            if not packet:
                break
            
            data_buffer += packet
            
            # Buscar los delimitadores JPEG (Inicio: \xff\xd8, Fin: \xff\xd9)
            start = data_buffer.find(b'\xff\xd8')
            end = data_buffer.find(b'\xff\xd9')
            
            if start != -1 and end != -1 and end > start:
                jpg_data = data_buffer[start:end+2]
                data_buffer = data_buffer[end+2:]
                
                # Decodificar imagen
                nparr = np.frombuffer(jpg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Si la cola está llena, saca el cuadro viejo y mete el nuevo
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    frame_queue.put(frame)

    except Exception as e:
        print(f"[RECEPTOR] Error en la red: {e}")
    finally:
        server_socket.close()

# Iniciar el hilo de recepción de red
hilo_red = threading.Thread(target=recibir_datos, daemon=True)
hilo_red.start()

# Bucle principal de la interfaz gráfica (Main Thread)
print("[RECEPTOR] Presiona 'q' en la ventana de video para salir.")

while running:
    if not frame_queue.empty():
        frame = frame_queue.get()
        cv2.imshow("Receptor de Video - Raspberry Pi", frame)
    else:
        # Pausa ligera para no saturar la CPU si no hay cuadros nuevos
        time.sleep(0.01)
    
    # Presionar 'q' para salir limpiamente
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cv2.destroyAllWindows()
print("[RECEPTOR] Programa finalizado limpiamente.")