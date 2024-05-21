import hashlib
import cv2
import socket
import mediapipe as mp
from cam_realsense_d400 import IntelRealSenseD435

stream_res_x = 640
stream_res_y = 480
fps = 30
preset_json = 'BodyScanPreset.json'

server_address = ("127.0.0.1", 5052)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_pose = mp.solutions.pose
pose_config = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def configureCamera():
    camera = IntelRealSenseD435(streamResX=stream_res_x, streamResY=stream_res_y, fps=fps, presetJSON=preset_json)
    camera.configureCamera()
    return camera

def getPoseLandmarks(image):
    return pose_config.process(image).pose_landmarks

# def calculateChecksum(data):
#     #data_encode = data.encode('utf-8')
#     return hashlib.md5(data.encode('UTF8')).hexdigest()

def calculateChecksum(data):
    # Calcula el hash SHA-256 en lugar de MD5
    sha256 = hashlib.sha256()
    sha256.update(data.encode('UTF8'))
    return sha256.hexdigest()

def verifyChecksum(data, checksum):
    # Verifica el hash SHA-256 en lugar de MD5
    calculated_checksum = calculateChecksum(data)
    isValid = calculated_checksum.lower() == checksum.lower()
    if not isValid:
        print("NOOOOOOOO")
        print(f"Verificación de checksum fallida. Calculado: {calculated_checksum}, Esperado: {checksum}")
    print("SIUUUUUUU")
    return isValid

def sendUDPMessage(sequence_number, data):
    data_str = ','.join(map(str, data))
    checksum = calculateChecksum(data_str)
    verifyChecksum(data_str, checksum)
    message = f"{sequence_number}:{data_str}:{checksum}\n"
    print(f"Checksum Leng: {len(checksum)}")
    print(f"Sending: {message}")  # Debugging line
    sock.sendto(message.encode(), server_address)

def drawPoseMarkersOnDepthImage(depth_image, markers, depth_scale):
    for landmark in markers.landmark[:11]:
        x = int(landmark.x * stream_res_x)
        y = stream_res_y - int(landmark.y * stream_res_y)
        if 0 <= y < stream_res_y and 0 <= x < stream_res_x:
            z = round(depth_image[y, x] * depth_scale, 2)
            cv2.circle(depth_image, (x, stream_res_y - y), 5, (0, 255, 255), -1)
            cv2.putText(depth_image, f'{z:.2f}', (x, stream_res_y - y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
                        x = int(landmark.x * stream_res_x)
                        y = stream_res_y - int(landmark.y * stream_res_y)
                        if 0 <= y < stream_res_y and 0 <= x < stream_res_x:
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
