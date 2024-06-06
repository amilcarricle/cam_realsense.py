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
#POSE_LANDMARKER_MODEL_PATH = "C:/Users/amilc/OneDrive/Escritorio/REpo/Tasks/pose_landmarker_lite.task"
POSE_LANDMARKER_MODEL_PATH = "C:/Users/amilc/OneDrive/Escritorio/REpo/model packages/pose_landmarker_full.task"
#POSE_LANDMARKER_MODEL_PATH = "C:/Users/amilc/OneDrive/Escritorio/REpo/model packages/pose_landmarker_heavy.task"

#Create a PoseLandmarker object
"""
ATTENTION!!!!!
Aclaracion, esta forma de crear los objetos difiere de los ejemplos mostrados por parte de google. Buscar en gitHub
un ejemplo donde se explica este cambio
"""
base_options = python.BaseOptions(model_asset_buffer=open(POSE_LANDMARKER_MODEL_PATH, "rb").read())
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

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

def correct_values(bookmarks, depth_image, depth_scale, color_image):
    """Corrects the depth values of the bookmarks."""
    if bookmarks:
        print("The number of values sent is verified")
        # Calculate initial depth values
        bookmarks[0].z = image_smoothing(bookmarks[0].x, bookmarks[0].y, depth_image, depth_scale)
        bookmarks[9].z = image_smoothing(bookmarks[9].x, bookmarks[9].y, depth_image, depth_scale)
        bookmarks[10].z = image_smoothing(bookmarks[10].x, bookmarks[10].y, depth_image, depth_scale)

        z_vals = [bookmarks[0].z, bookmarks[9].z, bookmarks[10].z]
        median_z = round(np.median(z_vals), 2)
        print(f"Median Z : {median_z}")

        # Right shoulder correction
        if bookmarks[11].z > 1.26 * median_z or bookmarks[11].z == 0 or bookmarks[11].z < median_z:
            displacement_x = 0
            flag = True
            while flag and displacement_x <= STREAM_RES_X:
                x = bookmarks[11].x - displacement_x
                bookmarks[11].z = image_smoothing(x, bookmarks[11].y, depth_image, depth_scale)
                if bookmarks[11].z > 1.26 * median_z or bookmarks[11].z == 0 or bookmarks[11].z < median_z:
                    displacement_x += 10
                    #print("AUN FUERA DE ESCALA")
                else:
                    flag = False
                    #print("EN ESCALA")
                # Check if displacement_x is within image bounds
                if x < bookmarks[12].x:
                    #print("Fuera de los límites de la imagen.")
                    break
        # Left shoulder correction
        if bookmarks[12].z > 1.26 * median_z or bookmarks[12].z == 0 or bookmarks[12].z < median_z:
            displacement_x = 0
            flag = True
            while flag and displacement_x <= STREAM_RES_X:
                x = bookmarks[12].x + displacement_x
                bookmarks[12].z = image_smoothing(x, bookmarks[12].y, depth_image, depth_scale)
                if bookmarks[12].z > 1.26 * median_z or bookmarks[12].z == 0 or bookmarks[12].z < median_z:
                    displacement_x += 10
                    #print("AUN FUERA DE ESCALA")
                else:
                    flag = False
                    #print("EN ESCALA")
                # Check if displacement_x is within image bounds
                if x > bookmarks[11].x:
                    #print("Fuera de los límites de la imagen.")
                    break

        #Left Elbow correction
        if bookmarks[13].z >= bookmarks[11].z and bookmarks[13].x <= STREAM_RES_X:
            flag = True
            print("HOLA")
            displacement_x = 1
            count = 0
            pixel= color_image[bookmarks[13].y, bookmarks[13].x]
            if (pixel == BG_COLOR).all():
                pixel_l = color_image[bookmarks[13].y + 30 + 10, bookmarks[13].x - 2]
                pixel_r = color_image[bookmarks[13].y - 30, bookmarks[13].x + 50]
                print(f"Pixel izquierdo : {pixel_l}")
                print(f"Pixel derecho : {pixel_r}")

            print(f"COLOR: {pixel}")

            midle_point = int(count / 2)
            # Calculate new depth value after finding the correct x coordinate
            bookmarks[13].z = image_smoothing(midle_point, bookmarks[13].y, depth_image,
                                              depth_scale)
        else:
            print("Elbow depth value within acceptable range.")
            #print(f"COLOR: {color_image[bookmarks[13].y, bookmarks[13].x]}")
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

                    bookmarks = correct_values(store_bookmarks, depth_image, depth_scale, color_image)
                    # print(f"Len of Bookmarks: {len(bookmarks)}")
                    sequence_number += 1
                    # Send UDP message with bookmarks
                    send_udp_message(sequence_number, [coord for bookmark in bookmarks for coord in
                                                       (bookmark.x, bookmark.y, bookmark.z)])
                    # Draw pose markers on depth image
                    draw_pose_markers_on_depth_image_from_bookmarks(depth_image, bookmarks)

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