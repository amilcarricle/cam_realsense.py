import cv2
import socket
import mediapipe as mp
from cam_realsense_d400 import IntelRealSenseD435

streamResX = 640
streamResY = 480
fps = 30
presetJSON = 'BodyScanPreset.json'

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddresPort = ("127.0.0.1", 5052)
mpDrawing = mp.solutions.drawing_utils
mpDarwingStyle = mp.solutions.drawing_styles
mpPose = mp.solutions.pose
poseConfig = mpPose.Pose(min_detection_confidence= 0.7, min_tracking_confidence= 0.7)

def main():
    # Configurar la cámara Intel RealSense D435
    realsense_camera = IntelRealSenseD435(streamResX=streamResX, streamResY=streamResY,
                                          fps=fps, presetJSON=presetJSON)
    #realsense_camera.configurePreset()
    realsense_camera.configureCamera()

    depthScale = realsense_camera.getDepthScale()
    print(depthScale)
    realsense_camera.startCapture()

    try:
        x_aux, y_aux, z_aux = None, None, None

        while True:
            color_image, depth_image = realsense_camera.getImageRGBDepth()
            storeBookmarks = []
            if color_image is not None:
                results = poseConfig.process(color_image)
                markers = results.pose_landmarks
                if markers:
                    for i in range (len(markers.landmark)):
                        x = int(markers.landmark[i].x * streamResX)
                        y = streamResY - int(markers.landmark[i].y * streamResY)
                        if 0 <= y < streamResY and 0 <= x < streamResX:
                            z = round(depth_image[y, x] * depthScale, 2)
                            if i == 0:
                                x_aux, y_aux, z_aux = x, y, z
                        else:
                            x, y, z =x_aux, y_aux, z_aux
                        #print(f'{i}: {x, y, z}')
                        storeBookmarks.append(x)
                        storeBookmarks.append(y)
                        storeBookmarks.append(z)
                    print(f'Markers List: {storeBookmarks}')
                    print(f'{len(storeBookmarks)}')
                    message = f"MARKERS:{storeBookmarks}"
                    sock.sendto(message.encode(), serverAddresPort)
                    sock.sendto(str.encode(str(storeBookmarks)), serverAddresPort)
                    print('Envio por puerto UDP')




                cv2.imshow('Color Image', color_image)
            if depth_image is not None:
                depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03  ), cv2.COLORMAP_JET)

                cv2.imshow('Depth Image', depth_image)

            # Esperar a que se presione la tecla 'q' para salir
            key = cv2.waitKey(1)
            if key == 27:
                break

    finally:
        # Detener la captura y cerrar las ventanas
        realsense_camera.stopCapture()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()