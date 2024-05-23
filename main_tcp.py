import hashlib
import cv2
import socket
import mediapipe as mp
import numpy as np
from cam_realsense_d400 import IntelRealSenseD435

# Camera configuration
STREAM_RES_X = 640
STREAM_RES_Y = 480
FPS = 30
PRESET_JSON = 'ShortRangePreset.json'

# UDP server configuration
SERVER_ADDRESS = ("127.0.0.1", 5052)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# MediaPipe Pose configuration
mp_pose = mp.solutions.pose
pose_config = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def configure_camera():
    """Configures and returns the Intel RealSense camera."""
    camera = IntelRealSenseD435(streamResX=STREAM_RES_X, streamResY=STREAM_RES_Y, fps=FPS, presetJSON=PRESET_JSON)
    camera.configureCamera()
    return camera

def get_pose_landmarks(image):
    """Gets the pose landmarks from a given image using MediaPipe."""
    return pose_config.process(image).pose_landmarks

def calculate_checksum(data):
    """Calculates and returns the SHA-256 checksum of the provided data."""
    sha256 = hashlib.sha256()
    sha256.update(data.encode('UTF8'))
    return sha256.hexdigest()

def verify_checksum(data, checksum):
    """Verifies that the calculated checksum matches the provided checksum."""
    calculated_checksum = calculate_checksum(data)
    is_valid = calculated_checksum.lower() == checksum.lower()
    if not is_valid:
        print(f"Checksum verification failed. Calculated: {calculated_checksum}, Expected: {checksum}")
    else:
        print("Checksum verification successful.")
    return is_valid

def send_udp_message(sequence_number, data):
    """Sends a UDP message with the data and its checksum."""
    data_str = ','.join(map(str, data))
    checksum = calculate_checksum(data_str)
    if verify_checksum(data_str, checksum):
        message = f"{sequence_number}:{data_str}:{checksum}\n"
        print(f"Sending: {message}")  # Debug line
        sock.sendto(message.encode(), SERVER_ADDRESS)
    else:
        print("Checksum verification failed, message not sent.")

def image_smoothing(point_x, point_y, depth_image, depth_scale):
    """Applies smoothing to the depth image at the specified point."""
    buffer = [(2, -2), (2, -1), (2, 0), (2, 1), (2, 2),
              (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
              (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
              (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
              (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2)]

    depth = []
    for dy, dx in buffer:
        ny, nx = point_y + dy, point_x + dx
        if 0 <= ny < STREAM_RES_Y and 0 <= nx < STREAM_RES_X:
            z = depth_image[ny, nx] * depth_scale
            depth.append(z)
    if depth:
        return round(np.median(depth), 2)
    return 0

def correct_values(bookmarks):
    """Corrects the depth values of the bookmarks."""
    if len(bookmarks) % 3 == 0:
        z_vals = [bookmarks[i*3 + 2] for i in range(11)]
        median_z = round(np.median(z_vals), 2)

        # Right arm correction
        if bookmarks[11*3 + 2] > median_z or bookmarks[11*3 + 2] == 0:
            bookmarks[11*3 + 2] = round(median_z * 1.13, 2)
        if bookmarks[11*3 + 2] > bookmarks[15*3 + 2] and bookmarks[13*3 + 2] == 0:
            bookmarks[13*3 + 2] = round((bookmarks[11*3 + 2] + bookmarks[15*3 + 2]) / 2, 2)
        if bookmarks[15*3 + 2] < bookmarks[11*3 + 2] and bookmarks[13*3 + 2] > bookmarks[11*3 + 2]:
            bookmarks[13*3 + 2] = round((bookmarks[11*3 + 2] + bookmarks[15*3 + 2]) / 2, 2)
        if bookmarks[15*3 + 2] > bookmarks[11*3 + 2] * 1.05:
            bookmarks[15*3 + 2] = round(bookmarks[13*3 + 2] * 0.3, 2)

        # Left arm correction
        if bookmarks[12*3 + 2] > median_z or bookmarks[12*3 + 2] == 0:
            bookmarks[12*3 + 2] = round(median_z * 1.13, 2)
        if bookmarks[12*3 + 2] > bookmarks[16*3 + 2] and bookmarks[14*3 + 2] == 0:
            bookmarks[14*3 + 2] = round((bookmarks[12*3 + 2] + bookmarks[16*3 + 2]) / 2, 2)
        if bookmarks[16*3 + 2] < bookmarks[12*3 + 2] and bookmarks[14*3 + 2] > bookmarks[12*3 + 2]:
            bookmarks[14*3 + 2] = round((bookmarks[12*3 + 2] + bookmarks[16*3 + 2]) / 2, 2)
        if bookmarks[16*3 + 2] > bookmarks[12*3 + 2] * 1.05:
            bookmarks[16*3 + 2] = round(bookmarks[14*3 + 2] * 0.3, 2)

    return bookmarks

def draw_pose_markers_on_depth_image_from_bookmarks(depth_image, bookmarks):
    """Draws pose markers on the depth image using a list of bookmarks."""
    for i in range(0, len(bookmarks), 3):
        x, y, z = bookmarks[i], bookmarks[i + 1], bookmarks[i + 2]
        if 0 <= y < STREAM_RES_Y and 0 <= x < STREAM_RES_X:
            cv2.circle(depth_image, (x, STREAM_RES_Y - y), 5, (255, 255, 255), -1)
            cv2.putText(depth_image, f'{z:.2f}', (x, STREAM_RES_Y - y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 125, 255), 2)

def draw_pose_markers_on_depth_image(depth_image, markers, depth_scale):
    """Draws pose markers on the depth image."""
    for landmark in markers.landmark[:17]:
        x = int(landmark.x * STREAM_RES_X)
        y = STREAM_RES_Y - int(landmark.y * STREAM_RES_Y)
        if 0 <= y < STREAM_RES_Y and 0 <= x < STREAM_RES_X:
            z = round(depth_image[y, x] * depth_scale, 2)
            cv2.circle(depth_image, (x, STREAM_RES_Y - y), 5, (0, 255, 255), -1)
            cv2.putText(depth_image, f'{z:.2f}', (x, STREAM_RES_Y - y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def main():
    sequence_number = 1
    realsense_camera = configure_camera()
    depth_scale = realsense_camera.getDepthScale()
    print(f'Depth Scale: {depth_scale}')
    realsense_camera.startCapture()

    try:
        x_aux, y_aux, z_aux = None, None, None

        while True:
            color_image, depth_image = realsense_camera.getImageRGBDepth()
            store_bookmarks = []

            if color_image is not None:
                markers = get_pose_landmarks(color_image)

                if markers:
                    for i, landmark in enumerate(markers.landmark[:17]):
                        x = int(landmark.x * STREAM_RES_X)
                        y = STREAM_RES_Y - int(landmark.y * STREAM_RES_Y)
                        if 0 <= y < STREAM_RES_Y and 0 <= x < STREAM_RES_X:
                            z = image_smoothing(x, y, depth_image, depth_scale)
                            x_aux, y_aux, z_aux = x, y, z
                        else:
                            x, y, z = x_aux, y_aux, z_aux
                        store_bookmarks.extend([x, y, z])
                    bookmarks = correct_values(store_bookmarks)
                    sequence_number += 1
                    send_udp_message(sequence_number, bookmarks)
                    draw_pose_markers_on_depth_image(depth_image, markers, depth_scale)
                    draw_pose_markers_on_depth_image_from_bookmarks(color_image, bookmarks)
                cv2.imshow('Color Image', color_image)
            if depth_image is not None:
                depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
                cv2.imshow('Depth Image', depth_image)

            if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        realsense_camera.stopCapture()
        cv2.destroyAllWindows()
        sock.close()

if __name__ == "__main__":
    main()
