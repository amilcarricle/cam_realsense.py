import numpy as np
import matplotlib.pyplot as plt


# Clase para representar un punto en 3D
class Punto3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Punto3D(x={self.x}, y={self.y}, z={self.z})"


# Función para cargar los datos de un archivo .txt y organizar los marcadores en una lista
def cargar_datos_txt(archivo):
    datos = []
    with open(archivo, 'r') as file:
        for line in file:
            # Convertir la línea en una lista de 39 elementos
            elementos = list(map(float, line.strip().split(',')))
            # Crear una lista de puntos 3D para esa fila
            puntos = [Punto3D(elementos[i], elementos[i + 1], elementos[i + 2]) for i in range(0, 39, 3)]
            datos.append(puntos)  # Añadir los 13 puntos (marcadores) a la lista de datos
    return datos


# Función para calcular la distancia entre dos puntos 3D
def calcular_distancia(punto1, punto2):
    return np.sqrt((punto1.x - punto2.x) ** 2 + (punto1.y - punto2.y) ** 2 + (punto1.z - punto2.z) ** 2)


# Función para calcular el ángulo usando el teorema del coseno
def calcular_angulo_teorema_coseno(punto1, punto2, punto3):
    # Distancias entre los puntos
    a = calcular_distancia(punto2, punto1)  # Distancia entre Hombro y Cadera
    b = calcular_distancia(punto3, punto2)  # Distancia entre Codo y Hombro
    c = calcular_distancia(punto3, punto1)  # Distancia entre Codo y Cadera

    # Aplicar el teorema del coseno
    cos_theta = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)

    # Asegurarse de que el valor de cos_theta esté entre -1 y 1 para evitar errores numéricos
    cos_theta = np.clip(cos_theta, -1, 1)

    # Calcular el ángulo en radianes y luego convertir a grados
    angulo = np.arccos(cos_theta)
    return np.degrees(angulo)


# Cargar los datos del archivo
datos = cargar_datos_txt('receivedData_3.txt')

# Lista para almacenar los ángulos del hombro en cada fila
angulos_hombro = []

# Calcular los ángulos del hombro para cada fila y almacenarlos
for fila in datos:
    cadera = fila[11]  # Marcador de la Cadera
    hombro = fila[5]  # Marcador del Hombro
    codo = fila[7]  # Marcador del Codo
    angulo = calcular_angulo_teorema_coseno(cadera, hombro, codo)
    angulos_hombro.append(angulo)

# Graficar los ángulos a lo largo de las filas (instantes de tiempo)
plt.plot(angulos_hombro, marker='o', linestyle='-', color='b')
plt.title("Ángulo del Hombro a lo Largo del Tiempo")
plt.xlabel("Instante de Tiempo (Filas de Datos)")
plt.ylabel("Ángulo del Hombro (Grados)")
plt.grid(True)
plt.show()
