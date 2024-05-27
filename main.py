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
mpPose = mp.solutions.pose
poseConfig = mpPose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def configure_camera():
    camera = IntelRealSenseD435(streamResX=streamResX, streamResY=streamResY, fps=fps, presetJSON=presetJSON)
    camera.configureCamera()
    return camera

def get_pose_landmarks(image):
    return poseConfig.process(image).pose_landmarks

def send_udp_message(data):
    message = f"MARKERS:{data}"
    sock.sendto(message.encode(), serverAddresPort)

def draw_pose_markers_on_depth_image(depth_image, markers, depth_scale):
    for landmark in markers.landmark[:11]:
        x = int(landmark.x * streamResX)
        y = streamResY - int(landmark.y * streamResY)
        if 0 <= y < streamResY and 0 <= x < streamResX:
            z = round(depth_image[y, x] * depth_scale, 2)
            cv2.circle(depth_image, (x, streamResY - y), 5, (0, 255, 255), -1)
            cv2.putText(depth_image, f'{z:.2f}', (x, streamResY - y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def main():
    realsense_camera = configure_camera()
    depthScale = realsense_camera.getDepthScale()
    print(f"Depth Scale: {depthScale}")
    realsense_camera.startCapture()

    try:
        x_aux, y_aux, z_aux = None, None, None

        while True:
            color_image, depth_image = realsense_camera.getImageRGBDepth()
            storeBookmarks = []

            if color_image is not None:
                markers = get_pose_landmarks(color_image)

                if markers:
                    for i, landmark in enumerate(markers.landmark[:11]):
                        x = int(landmark.x * streamResX)
                        y = streamResY - int(landmark.y * streamResY)
                        if 0 <= y < streamResY and 0 <= x < streamResX:
                            z = round(depth_image[y, x] * depthScale, 2)
                            x_aux, y_aux, z_aux = x, y, z
                        else:
                            x, y, z = x_aux, y_aux, z_aux
                        storeBookmarks.extend([x, y, z])

                    print(f'Markers List: {storeBookmarks}')
                    print(f'Number of Markers: {len(storeBookmarks)}')
                    send_udp_message(storeBookmarks)
                    print('Sent UDP message')

                    # Draw pose markers on depth image
                    draw_pose_markers_on_depth_image(depth_image, markers, depthScale)

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

if __name__ == "__main__":
    main()
