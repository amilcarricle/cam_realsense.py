import hashlib
import cv2
import socket
import mediapipe as mp
from cam_realsense_d400 import IntelRealSenseD435  # Asegúrate de tener esta clase implementada

stream_res_x = 640
stream_res_y = 480
fps = 30
preset_json = 'BodyScanPreset.json'
sequence_numbers = 0
server_address = ("127.0.0.1", 5052)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_pose = mp.solutions.pose
pose_config = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def configureCamera():
    camera = IntelRealSenseD435(streamResX=stream_res_x, streamResY=stream_res_y,
                                fps=fps, presetJSON=preset_json)
    camera.configureCamera()
    return camera

def getPoseLandmarks(image):
    return pose_config.process(image).pose_landmarks

def calculateChecksum(data):
    return hashlib.md5(data.encode()).hexdigest()
def sendUDPMessage(sequence_numbers, data):
    data_str = ','.join(map(str, data))
    checksum = calculateChecksum(data_str)
    message = f"{sequence_numbers}:{data_str}:{checksum}\n"
    sock.sendto(message.encode(), server_address)

def drawPoseMarkersOnDepthImage(depth_image, markers, depth_scale):
    for landmark in markers.landmark[:11]:
        x = int(landmark.x * stream_res_x)
        y = stream_res_y - int(landmark.y * stream_res_y)
        if 0 <= y < stream_res_y and 0 <= x < stream_res_x:
            z = round(depth_image[y, x] * depth_scale, 2)
            cv2.circle(depth_image, (x, stream_res_y - y), 5, (0, 255, 255), -1)
            cv2.putText(depth_image, f'{z:.2f}', (x, stream_res_y - y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def main():
    global sequence_numbers
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
                    sequence_numbers += 1
                    print(f'Sequence {sequence_numbers} - Markers List: {store_bookmarks}')
                    print(f'Number of Markers: {len(store_bookmarks)}')
                    print(f'Markers List: {store_bookmarks}')
                    print(f'Number of Markers: {len(store_bookmarks)}')
                    sendUDPMessage(sequence_numbers, store_bookmarks)
                    print('Sent UDP message')

                    # Draw pose markers on depth image
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
