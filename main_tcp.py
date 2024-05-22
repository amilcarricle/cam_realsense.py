import hashlib
import cv2
import socket
import mediapipe as mp
from cam_realsense_d400 import IntelRealSenseD435

# Configuración de la cámara
STREAM_RES_X = 640
STREAM_RES_Y = 480
FPS = 30
PRESET_JSON = 'BodyScanPreset.json'

# Configuración del servidor UDP
SERVER_ADDRESS = ("127.0.0.1", 5052)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Configuración de MediaPipe Pose
mp_pose = mp.solutions.pose
pose_config = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def configureCamera():
    camera = IntelRealSenseD435(streamResX=STREAM_RES_X, streamResY=STREAM_RES_Y, fps=FPS, presetJSON=PRESET_JSON)
    camera.configureCamera()
    return camera

def getPoseLandmarks(image):
    return pose_config.process(image).pose_landmarks

def calculateChecksum(data):
    sha256 = hashlib.sha256()
    sha256.update(data.encode('UTF8'))
    return sha256.hexdigest()

def verifyChecksum(data, checksum):
    calculated_checksum = calculateChecksum(data)
    isValid = calculated_checksum.lower() == checksum.lower()
    if not isValid:
        print("NOOOOOOOO")
        print(f"Verificación de checksum fallida. Calculado: {calculated_checksum}, Esperado: {checksum}")
    else:
        print("SIUUUUUUU")
    return isValid

def sendUDPMessage(sequence_number, data):
    data_str = ','.join(map(str, data))
    checksum = calculateChecksum(data_str)
    if verifyChecksum(data_str, checksum):
        message = f"{sequence_number}:{data_str}:{checksum}\n"
        print(f"Sending: {message}")  # Línea de depuración
        sock.sendto(message.encode(), SERVER_ADDRESS)
    else:
        print("Checksum verification failed, not sending the message.")

def drawPoseMarkersOnDepthImage(depth_image, markers, depth_scale):
    for landmark in markers.landmark[:11]:
        x = int(landmark.x * STREAM_RES_X)
        y = STREAM_RES_Y - int(landmark.y * STREAM_RES_Y)
        if 0 <= y < STREAM_RES_Y and 0 <= x < STREAM_RES_X:
            z = round(depth_image[y, x] * depth_scale, 2)
            cv2.circle(depth_image, (x, STREAM_RES_Y - y), 5, (0, 255, 255), -1)
            cv2.putText(depth_image, f'{z:.2f}', (x, STREAM_RES_Y - y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def main():
    global sequence_number
    sequence_number = 1
    realsense_camera = configureCamera()
    depth_scale = realsense_camera.getDepthScale()
    print(f'Depth Scale: {depth_scale}')
    realsense_camera.startCapture()

    try:
        x_aux, y_aux, z_aux = None, None, None

        while True:
            color_image, depth_image = realsense_camera.getImageRGBDepth()
            store_bookmarks = []

            if color_image is not None:
                markers = getPoseLandmarks(color_image)

                if markers:
                    for i, landmark in enumerate(markers.landmark[:11]):
                        x = int(landmark.x * STREAM_RES_X)
                        y = STREAM_RES_Y - int(landmark.y * STREAM_RES_Y)
                        if 0 <= y < STREAM_RES_Y and 0 <= x < STREAM_RES_X:
                            z = round(depth_image[y, x] * depth_scale, 2)
                            x_aux, y_aux, z_aux = x, y, z
                        else:
                            x, y, z = x_aux, y_aux, z_aux
                        store_bookmarks.extend([x, y, z])
                    sequence_number += 1
                    sendUDPMessage(sequence_number, store_bookmarks)
                    drawPoseMarkersOnDepthImage(depth_image, markers, depth_scale)

                cv2.imshow('Color Image', color_image)
            if depth_image is not None:
                depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
                cv2.imshow('Depth Image', depth_image)

            if cv2.waitKey(1) == 27:  # Presionar 'Esc' para salir
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        realsense_camera.stopCapture()
        cv2.destroyAllWindows()
        sock.close()

if __name__ == "__main__":
    main()
