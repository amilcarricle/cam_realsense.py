import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


# Función para graficar la trayectoria del marcador del hombro en 3D y en los diferentes planos usando las filas de datos
def graficar_trayectoria_hombro_vs_datos(datos):
    # Inicializar listas para almacenar las coordenadas del hombro y los índices de las filas
    x_hombro = []
    y_hombro = []
    z_hombro = []
    filas = list(range(len(datos)))  # Lista de índices (filas de datos)

    # Extraer los datos del hombro en cada fila
    for fila in datos:
        hombro = fila[5]  # Suponemos que el hombro es el primer marcador
        x_hombro.append(hombro.x)
        y_hombro.append(hombro.y)
        z_hombro.append(hombro.z)

    # Crear la figura
    fig = plt.figure(figsize=(10, 10))

    # Subplot para la trayectoria en 3D (Filas de Datos vs X, Y, Z)
    """
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(filas, x_hombro, z_hombro, marker='o', color='b')
    ax1.set_title('Trayectoria del Hombro en 3D (Filas de Datos vs X y Z)')
    ax1.set_xlabel('Filas de Datos')
    ax1.set_ylabel('X (ancho del cuerpo)')
    ax1.set_zlabel('Z (distancia a la cámara)')
    ax1.grid(True)

    # Subplot para el plano frontal (coronal): X vs Z vs Filas de Datos
    ax2 = fig.add_subplot(222)
    ax2.plot(filas, x_hombro, marker='o', color='r')
    ax2.set_title('Plano Frontal (Coronal) - Filas de Datos vs X')
    ax2.set_xlabel('Filas de Datos')
    ax2.set_ylabel('X (ancho del cuerpo)')
    ax2.grid(True)
    
    # Subplot para el plano sagital: Y vs Z vs Filas de Datos
    ax3 = fig.add_subplot(223)
    ax3.plot(filas, y_hombro, marker='o', color='g')
    ax3.set_title('Plano Sagital - Filas de Datos vs Y')
    ax3.set_xlabel('Filas de Datos')
    ax3.set_ylabel('Y (altura del cuerpo)')
    ax3.grid(True)
    """
    # Subplot para el plano transversal: Z vs Filas de Datos
    ax4 = fig.add_subplot(111)
    ax4.plot(filas, z_hombro, marker='o', color='b')
    ax4.set_title('Plano Transversal - Filas de Datos vs Z')
    ax4.set_xlabel('Filas de Datos')
    ax4.set_ylabel('Z (distancia a la cámara)')
    ax4.grid(True)

    # Mostrar los gráficos
    plt.tight_layout()
    plt.show()


# Cargar los datos del archivo
datos = cargar_datos_txt('receivedData_4.txt')

# Graficar la trayectoria del marcador del hombro vs filas de datos
graficar_trayectoria_hombro_vs_datos(datos)
