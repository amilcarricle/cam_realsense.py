import cv2

from cam_realsense import IntelRealSenseD435

def main():
    # Configurar la cámara Intel RealSense D435
    realsense_camera = IntelRealSenseD435(streamResX=640, streamResY=480, fps=30)
    realsense_camera.configureCamera()
    realsense_camera.startCapture()

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