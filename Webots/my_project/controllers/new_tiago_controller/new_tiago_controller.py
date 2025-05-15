from controller import Robot, Camera, Lidar
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import csv

# Imágenes de objetos
image_counter = 0
if not os.path.exists("detecciones"):
    os.makedirs("detecciones")
    
# Crear CSV
csv_filename = "detecciones.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "image_path", "object", "confidence", "x1", "y1", "x2", "y2"])
    
# Inicializar el robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Motores de ruedas
left_motor = robot.getDevice("wheel_left_joint")
right_motor = robot.getDevice("wheel_right_joint")
left_motor.setPosition(float('inf'))  
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)  
right_motor.setVelocity(0.0)

# Sensores de distancia (LiDAR)
lidar = robot.getDevice("Hokuyo URG-04LX-UG01")  
lidar.enable(timestep)
lidar.enablePointCloud()

# Cámara
camera = robot.getDevice("Astra rgb")
camera.enable(timestep)

# Cargar modelo YOLO
model = YOLO("yolov8n.pt")

# Motores del brazo
arm_joints = []
for i in range(1, 8):  
    arm_joints.append(robot.getDevice(f"arm_{i}_joint"))

# Definir una posición en alto para el brazo
arm_positions = [0.07, 0.26, -3.16, 1.27, 1.32, 0.0, 1.41]
for i, joint in enumerate(arm_joints):
    joint.setPosition(arm_positions[i])

# Parámetros de velocidad
MAX_SPEED = 3.0  
THRESHOLD = 1.5  

# Bucle principal
while robot.step(timestep) != -1:
    # Obtener solo datos de la parte frontal
    front_data = lidar.getRangeImage()[len(lidar.getRangeImage())//3 : 2*len(lidar.getRangeImage())//3]
    
    # Filtrar valores inválidos
    lidar_filtered = [d for d in front_data if d > 0.1 and d < 10.0]

    # Detectar obstáculos correctamente
    if lidar_filtered:
        min_distance = min(lidar_filtered)
    else:
        min_distance = 999  # No hay obstáculos

    # Evitar colisiones
    if min_distance < THRESHOLD:
        print("Obstáculo detectado. Girando...")
        left_motor.setVelocity(-MAX_SPEED)
        right_motor.setVelocity(MAX_SPEED)  
    else:
        left_motor.setVelocity(MAX_SPEED)
        right_motor.setVelocity(MAX_SPEED)

    # Captura de imagen
    image = camera.getImage()
    if image:

        # Convertir la imagen de Webots a OpenCV
        width, height = camera.getWidth(), camera.getHeight()
        img_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))  # Webots usa RGBA
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)  # Convertir a BGR para OpenCV

        # Detección de objetos con YOLO
        results = model(img_bgr)

        deteccion_realizada = False

        # Dibujar las detecciones en la imagen
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                # Dibujar la caja y etiqueta en la imagen
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_bgr, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"🟢 Objeto detectado: {label} con {confidence:.2f} de confianza")
                deteccion_realizada = True

        # Guardar imagen si se detectó algún objeto
        if deteccion_realizada:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_filename = f"detecciones/deteccion_{timestamp}.png"
            cv2.imwrite(image_filename, img_bgr)
            print(f"💾 Imagen guardada: {image_filename}")

            # Guardar información en el CSV
            with open(csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)

                # Guardar cada detección en una fila
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = result.names[int(box.cls[0])]
                        confidence = float(box.conf[0])
                        
                        writer.writerow([timestamp, image_filename, label, confidence, x1, y1, x2, y2])

        # Mostrar la imagen con detecciones
        cv2.imshow("YOLO Detections", img_bgr)
        cv2.waitKey(1)  # Necesario para actualizar la ventana en tiempo real