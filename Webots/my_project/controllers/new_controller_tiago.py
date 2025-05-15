from controller import Robot, Camera, Lidar

# Inicializar el robot
robot = Robot()

# Obtener timestep del simulador
timestep = int(robot.getBasicTimeStep())

# Motores de ruedas
left_motor = robot.getDevice("wheel_left_joint")
right_motor = robot.getDevice("wheel_right_joint")
left_motor.setPosition(float('inf'))  # Modo de velocidad
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)  # Inicializar velocidad
right_motor.setVelocity(0.0)

# Sensores de distancia (LiDAR)
lidar = robot.getDevice("lidar_2d")  # Nombre del LiDAR en Webots
lidar.enable(timestep)
lidar.enablePointCloud()

# Cámara
camera = robot.getDevice("camera")
camera.enable(timestep)

# Parámetros de velocidad
MAX_SPEED = 2.0  # Velocidad máxima del robot
THRESHOLD = 0.5  # Umbral de distancia para evitar colisiones

# Bucle principal
while robot.step(timestep) != -1:
    # Obtener datos del LiDAR
    lidar_data = lidar.getRangeImage()
    
    # Evitar colisiones (simple detección frontal)
    min_distance = min(lidar_data)  # Distancia mínima detectada
    if min_distance < THRESHOLD:
        print("Obstáculo detectado. Girando...")
        left_motor.setVelocity(-MAX_SPEED)
        right_motor.setVelocity(MAX_SPEED)  # Giro a la derecha
    else:
        left_motor.setVelocity(MAX_SPEED)
        right_motor.setVelocity(MAX_SPEED)

    # Captura de imagen de la cámara
    image = camera.getImage()
    
    if image:
        print("Imagen capturada.")

