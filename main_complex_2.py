import hashlib
import cv2
import socket
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from cam_realsense_d400 import IntelRealSenseD435
from dataclasses import dataclass

@dataclass
class Bookmark:
    x: int
    y: int
    z: float

# Camera configuration
STREAM_RES_X = 640
STREAM_RES_Y = 480
FPS = 30
PRESET_JSON = 'ShortRangePreset.json'

#Segmentation mask
BG_COLOR = (125, 125, 125)
MASK_COLOR = (255, 255, 255)

# UDP server configuration
SERVER_ADDRESS = ("127.0.0.1", 5052)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# MediaPipe Pose configuration
#Model Path: Model selection (Lite, Full, Heavy)
#POSE_LANDMARKER_MODEL_PATH = "C:/Users/Ricle/Desktop/Amilcar/CamRSManager/model packages/pose_landmarker_lite.task"
#POSE_LANDMARKER_MODEL_PATH = "C:/Users/Ricle/Desktop/Amilcar/CamRSManager/model packages/pose_landmarker_full.task"
POSE_LANDMARKER_MODEL_PATH = "C:/Users/Ricle/Desktop/Amilcar/CamRSManager/model packages/pose_landmarker_heavy.task"

#Create a PoseLandmarker object
"""
ATTENTION!!!!!
Aclaracion, esta forma de crear los objetos difiere de los ejemplos mostrados por parte de google. Buscar en gitHub
un ejemplo donde se explica este cambio
"""
base_options = python.BaseOptions(model_asset_buffer=open(POSE_LANDMARKER_MODEL_PATH, "rb").read())
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)
# base_options = python.BaseOptions(model_asset_path =POSE_LANDMARKER_MODEL_PATH)
# options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
# detector = vision.PoseLandmarker.create_from_options(options)

def configure_camera():
    """Configures and returns the Intel RealSense camera."""
    camera = IntelRealSenseD435(streamResX=STREAM_RES_X, streamResY=STREAM_RES_Y,
                                fps=FPS, presetJSON=PRESET_JSON)
    camera.configureCamera()
    return camera

def get_pose_landmarks(image):
    """Gets the pose landmarks from a given image using MediaPipe."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = detector.detect(mp_image)
    return result.pose_landmarks, result.segmentation_masks

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
        print(f"Checksum verification failed. Calculated: {calculated_checksum}, "
              f"Expected: {checksum}")
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
    buffer_size = 8
    depth_values = depth_image[max(0, point_y - buffer_size):
                               min(STREAM_RES_Y, point_y + buffer_size + 1),
                               max(0, point_x - buffer_size):
                               min(STREAM_RES_X, point_x + buffer_size + 1)]
    z_values = depth_values.flatten() * depth_scale
    return round(np.median(z_values), 2)

def calculate_distance(x1, y1, x2, y2):
    """Calculates the Euclidean distance between two points."""
    return round(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

def correct_values(bookmarks):
    """Corrects the depth values of the bookmarks."""
    if len(bookmarks) == 33:
        print("The number of values sent is verified")
        z_vals = [bookmark.z for bookmark in bookmarks[:11]]
        median_z = round(np.median(z_vals), 2)
        print(f"Median Z : {median_z}")
        # Right shoulder correction
        if bookmarks[11].z > 1.26 * median_z or bookmarks[11].z == 0:
            bookmarks[11].z = round(median_z * 1.13, 2)
        # Left shoulder correction
        if bookmarks[12].z > 1.26 * median_z or bookmarks[12].z == 0:
            bookmarks[12].z = round(median_z * 1.13, 2)
        # When, during the flexion of the arm, the hand overlaps the shoulder,
        # the marker changes its distance to the wrong one.
        if bookmarks[13].z < 0.4 * bookmarks[11].z and bookmarks[11].z != 0 and bookmarks[12].z != 0 and (bookmarks[11].z / bookmarks[12].z) < 0.7:
            bookmarks[11].z = bookmarks[12].z
            bookmarks[13].z = round(0.6 * bookmarks[11].z, 2)
        if bookmarks[13].z == 0:
            bookmarks[13].z = round(0.6 * bookmarks[11].z, 2)
        if bookmarks[13].z > 1.2 * median_z:
            bookmarks[13].z = round(median_z * 1.13, 2)
        if bookmarks[15].z == 0:
            bookmarks[15].z = round(bookmarks[11].z * 0.5, 2)
        if bookmarks[15].z > 1.2 * median_z:
            bookmarks[15].z = round(median_z * 1.13, 2)
        if bookmarks[15].z > bookmarks[13].z and bookmarks[13].z != 0:
            bookmarks[15].z = round(bookmarks[11].z * 0.2, 2)

        if bookmarks[14].z < 0.4 * bookmarks[12].z and bookmarks[12].z != 0 and bookmarks[11].z != 0 and (bookmarks[12].z / bookmarks[11].z) < 0.7:
            bookmarks[12].z = bookmarks[11].z
            bookmarks[14].z = round(0.6 * bookmarks[12].z, 2)
        if bookmarks[14].z == 0:
            bookmarks[14].z = round(0.6 * bookmarks[12].z, 2)
        if bookmarks[14].z > 1.2 * median_z:
            bookmarks[14].z = round(median_z * 1.13, 2)
        if bookmarks[16].z == 0:
            bookmarks[16].z = round(bookmarks[12].z * 0.5, 2)
        if bookmarks[16].z > 1.2 * median_z:
            bookmarks[16].z = round(median_z * 1.13, 2)
        if bookmarks[16].z > bookmarks[14].z and bookmarks[14].z != 0:
            bookmarks[16].z = round(bookmarks[12].z * 0.2, 2)
    else:
        print("The number of values sent is incorrect")
    return bookmarks

def draw_pose_markers_on_depth_image_from_bookmarks(depth_image, bookmarks):
    """Draws pose markers on the depth image using a list of bookmarks."""
    for bookmark in bookmarks:
        x, y, z = bookmark.x, bookmark.y, bookmark.z
        if 0 <= y < STREAM_RES_Y and 0 <= x < STREAM_RES_X:
            x = int(x)
            y = int(y)
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
            cv2.putText(depth_image, f'{z:.2f}', (x, STREAM_RES_Y - y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 125, 255), 2)


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
                pose_landmarks_list, segmentation_mask = get_pose_landmarks(color_image)

                if pose_landmarks_list:
                    #print(f"Markers: {pose_landmarks_list}")
                    for markers in pose_landmarks_list:
                        for idx, landmark in enumerate(markers):
                            x = int(landmark.x * STREAM_RES_X)
                            y = STREAM_RES_Y - int(landmark.y * STREAM_RES_Y)
                            if 0 <= y < STREAM_RES_Y and 0 <= x < STREAM_RES_X:
                                #z = image_smoothing(x, y, depth_image, depth_scale)
                                z = round(depth_image[y, x] * depth_scale, 2)
                                x_aux, y_aux, z_aux = x, y, z
                            else:
                                x, y, z = x_aux, y_aux, z_aux
                            store_bookmarks.append(Bookmark(x, y, z))

                            # Print the coordinates
                            #print(f"Landmark {idx}: (x={x}, y={y}, z={z})")

                    bookmarks = correct_values(store_bookmarks)
                    #print(f"Len of Bookmarks: {len(bookmarks)}")
                    sequence_number += 1
                    # Send UDP message with bookmarks
                    send_udp_message(sequence_number, [coord for bookmark in bookmarks for coord in
                                                       (bookmark.x, bookmark.y, bookmark.z)])
                    # Draw pose markers on depth image
                    draw_pose_markers_on_depth_image_from_bookmarks(depth_image, bookmarks)
                #Apply segmentation mask to color image
                if segmentation_mask is not None:
                    mask = np.asarray(segmentation_mask[0].numpy_view())
                    mask = cv2.resize(mask, (STREAM_RES_X, STREAM_RES_Y))
                    mask = (mask * 255).astype(np.uint8)  # Ensure mask is in uint8 format
                    mask = np.stack((mask,) * 3, axis=-1)
                    mask = cv2.bitwise_and(mask, np.array(MASK_COLOR, dtype=np.uint8))

                    fg_image = cv2.bitwise_and(color_image, mask)
                    bg_image = np.zeros(color_image.shape, dtype=np.uint8)
                    bg_image[:] = BG_COLOR
                    bg_image = cv2.bitwise_and(bg_image, cv2.bitwise_not(mask))
                    color_image = cv2.add(fg_image, bg_image)

                cv2.imshow('Color Image', color_image)

            if depth_image is not None:
                # Apply color map and show depth image
                depth_image_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
                cv2.imshow('Depth Image', depth_image_colored)

            # Exit if 'Esc' key is pressed
            if cv2.waitKey(1) == 27:
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop camera capture, close OpenCV windows, and close socket
        realsense_camera.stopCapture()
        cv2.destroyAllWindows()
        sock.close()


if __name__ == "__main__":
    main()

