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

def cargar_datos_txt(archivo):
    datos = []
    with open(archivo, 'r') as file:
        for line in file:
            elementos = list(map(float, line.strip().split(',')))
            puntos = [Punto3D(elementos[i], elementos[i + 1], elementos[i + 2]) for i in range(0, 39, 3)]
            datos.append(puntos)
    return datos

def calcular_distancia(punto_1, punto_2):
    return np.sqrt((punto_1.x - punto_2.x) ** 2 + (punto_1.y - punto_2.y) ** 2 + (punto_1.z - punto_2.z) ** 2)

def calcular_angulos(marcador_1, marcador_2, marcador_3):
    a = calcular_distancia(marcador_1, marcador_2)
    b = calcular_distancia(marcador_3, marcador_2)
    c = calcular_distancia(marcador_3, marcador_1)

    theta = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    theta = np.clip(theta, -1, 1)  # Asegurar que theta esté entre -1 y 1
    angulo = np.arccos(theta)
    return np.degrees(angulo)

def determinar_maximos(angulos, ventana_size, angulo_inferior, angulo_superior):
    maximos = []
    indices_maximos = []
    for i in range(0, len(angulos) - ventana_size + 1, ventana_size):
        ventana = angulos[i:i + ventana_size]
        ventana_filtrada = [angulo for angulo in ventana if angulo_inferior <= angulo <= angulo_superior]
        if ventana_filtrada:
            maximo = np.max(ventana_filtrada)
            maximos.append(maximo)
            indices_maximos.append(i + np.argmax(ventana_filtrada))
        else:
            maximos.append(None)
            indices_maximos.append(None)

    return maximos, indices_maximos

def calcular_velocidad_aceleracion(z, tiempos):
    velocidad = np.gradient(z, tiempos)
    aceleracion = np.gradient(velocidad, tiempos)
    return velocidad, aceleracion

def main():
    angulos_hombro = []
    posicion_z = []
    datos = cargar_datos_txt('marcadores_corregidos.txt')
    contador_evento_1 = contador_evento_2 = 0
    for marcador in datos:
        hombro_izq = marcador[6]
        hombro = marcador[5]
        codo = marcador[7]
        codo_izq = marcador[8]
        munieca = marcador[9]
        cadera = marcador[11]

        dist_1 = calcular_distancia(hombro_izq, munieca)
        dist_2 = calcular_distancia(codo_izq, munieca)
        angulo = calcular_angulos(cadera, hombro, codo)

        if dist_1 <= 50 and 60 <= angulo <= 90:
            contador_evento_1 += 1
        if dist_2 <= 50 and 20 <= angulo < 40:
            contador_evento_2 += 1

        angulos_hombro.append(angulo)
        posicion_z.append(hombro.z)

    duracion_total = len(angulos_hombro) / 30
    tiempos = np.linspace(0, duracion_total, len(angulos_hombro))

    limite_inferior = 120
    limite_superior = 140
    ventana = 100
    maximos_valores, indices_maximos = determinar_maximos(angulos_hombro, ventana, limite_inferior, limite_superior)

    tiempos_maximos = [tiempos[i] for i in indices_maximos if i is not None]
    maximos_valores_filtrados = [m for m, i in zip(maximos_valores, indices_maximos) if i is not None]

    velocidad_z, aceleracion_z = calcular_velocidad_aceleracion(posicion_z, tiempos)

    print("Máximos encontrados en las ventanas:", maximos_valores_filtrados)
    print(f"{len(maximos_valores_filtrados)}")

    print(f"Eventos HombroIzq - MuniecaDer: {contador_evento_1}")
    print(f"Eventos CodoIzq - MuniecaDer: {contador_evento_2}")

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(tiempos, angulos_hombro, marker='o', linestyle='-', color='g', label='Ángulo del Hombro')
    axs[0].scatter(tiempos_maximos, maximos_valores_filtrados, color='blue', marker='x', s=100, label='Máximo en Ventana')
    axs[0].set_title("Ángulo de Hombro a lo largo del tiempo")
    axs[0].set_ylabel("Ángulos del Hombro [°]")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(tiempos, posicion_z, color='r', label='Posición Z')
    axs[1].set_title("Posición Z a lo largo del tiempo")
    axs[1].set_ylabel("Posición Z [m]")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(tiempos, velocidad_z, color='b', label='Velocidad Z')
    axs[2].set_title("Velocidad Z a lo largo del tiempo")
    axs[2].set_ylabel("Velocidad Z [m/s]")
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(tiempos, aceleracion_z, color='m', label='Aceleración Z')
    axs[3].set_title("Aceleración Z a lo largo del tiempo")
    axs[3].set_xlabel("Tiempo [s]")
    axs[3].set_ylabel("Aceleración Z [m/s²]")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
