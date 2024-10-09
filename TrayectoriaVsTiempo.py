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

def cargar_datos_txt(archivo):
    """
    :param archivo:
    :return: datos
    """
    datos = []
    with open(archivo, 'r') as file:
        for line in file:
            elementos = list(map(float, line.strip().split(',')))
            puntos = [Punto3D(elementos[i], elementos[i + 1], elementos[i + 2]) for i in range(0, 39, 3)]
            datos.append(puntos)
    return datos

def calcular_distancia(punto_1, punto_2):
    """
    :param punto_1:
    :param punto_2:
    :return: distancia euclidea
    """
    return np.sqrt((punto_1.x - punto_2.x) ** 2 + (punto_1.y - punto_2.y) ** 2 + (punto_1.z - punto_2.z) ** 2)

def calcular_angulos(marcador_1, marcador_2, marcador_3):
    a = calcular_distancia(marcador_1, marcador_2)
    b = calcular_distancia(marcador_3, marcador_2)
    c = calcular_distancia(marcador_3, marcador_1)

    theta = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    #Aseguro que theta este entre valores de 1 y -1
    theta = np.clip(theta, -1,1)
    angulo = np.arccos(theta)
    return np.degrees(angulo)
def main():
    angulos_hombro = []
    datos = cargar_datos_txt('fulanito_2.txt')

    for marcador in datos:
        hombro = marcador[5]
        codo = marcador[7]
        cadera = marcador[11]
        angulo = calcular_angulos(cadera, hombro, codo)
        angulos_hombro.append(angulo)
    print(f"{len(datos)}")
    #print(f"{len(hombro)}")
    print(f"{len(angulos_hombro)}")
    duracion_total = len(angulos_hombro) / 30  # Duración total en segundos
    tiempos = np.linspace(0, duracion_total * 5.86, len(angulos_hombro))
    print(f"{len(angulos_hombro) * 5.86 / 30}")
    plt.plot(tiempos, angulos_hombro, marker = 'o', linestyle = '-', color = 'g')
    plt.title("Angulo de Hombro a lo largo del tiempo")
    plt.ylabel("Angulos del Hombro [°]")
    plt.xlabel("Datos")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()