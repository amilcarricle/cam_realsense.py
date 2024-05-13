import cv2
import os
from cam_realsense import IntelRealSenseD435

def main():
    # Configurar la cámara Intel RealSense D435
    preset_json = "EdgeMapD435.json"
    print(os.getcwd(),preset_json)
    json_path = os.path.join(os.getcwd(), preset_json)
    realsense_camera = IntelRealSenseD435(640, 480, 30)
    realsense_camera.configureCamera()
    realsense_camera.startCapture()

    try:
        with open(json_path) as preset_json:
            # Procesa el archivo .json aquí
            pass
    except FileNotFoundError:
        print(f"El archivo {json_path} no se encuentra en el directorio actual.")

    try:

        while True:
            # Obtener imágenes RGB y de profundidad
            color_image, depth_image = realsense_camera.getImageRGBDepth()

            # Mostrar las imágenes en ventanas separadas
            if color_image is not None:
                cv2.imshow('Color Image', color_image)
            if depth_image is not None:
                cv2.imshow('Depth Image', depth_image)

            # Esperar a que se presione la tecla 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Detener la captura y cerrar las ventanas
        realsense_camera.stopCapture()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()