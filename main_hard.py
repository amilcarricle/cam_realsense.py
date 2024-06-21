import hashlib
import cv2
import socket
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cam_realsense_d400 import IntelRealSenseD435
from dataclasses import dataclass
from math import acos, degrees
@dataclass
class Bookmark:
    x: int
    y: int
    z: float

# Anatomic distance
DIST_SHOULDER_ELBOW = 0.30
DIST_ELBOW_WRIST = 0.25
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
POSE_LANDMARKER_MODEL_PATH = "pose_landmarker_full.task"
#POSE_LANDMARKER_MODEL_PATH = "pose_landmarker_heavy.task"

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

def is_nearby(marker1, marker2, threshold=30):
    """Checks if two markers are within a certain distance threshold."""
    distance = np.sqrt((marker1.x - marker2.x) ** 2 + (marker1.y - marker2.y) ** 2)
    return distance < threshold
# Function to check if points are aligned in a straight line (indicating full extension)
def angle_arm(shoulder, elbow, wrist):
    p1 = np.array([shoulder.x, shoulder.y])
    p2 = np.array([elbow.x, elbow.y])
    p3 = np.array([wrist.x, wrist.y])

    l1 = np.linalg.norm(p2 - p3)
    l2 = np.linalg.norm(p1 - p3)
    l3 = np.linalg.norm(p1 - p2)

    return (degrees ( acos( (l1 ** 2 + l3 ** 2 - l2 ** 2) / (2 * l1 * l3) ) ) )

#Shoulder correction
def shoulder_correction(shoulder, shoulder_lim, median_z, depth_image, depth_scale):
    flag = True
    if shoulder.z > 1.26 * median_z or shoulder.z == 0 or shoulder.z < median_z:
        displacement_x = 0
        #For right shoulder
        if shoulder_lim.x < shoulder.x:
            while flag and displacement_x <= STREAM_RES_X:
                x = shoulder.x - displacement_x
                shoulder.z = image_smoothing(x, shoulder.y, depth_image, depth_scale)
                if shoulder.z > 1.26 * median_z or shoulder.z == 0 or shoulder.z < median_z:
                    displacement_x += 10
                else:
                    flag = False
                if x < shoulder_lim.x:
                    break
        #For left shoulder
        if shoulder_lim.x > shoulder.x:
            while flag and displacement_x <= STREAM_RES_X:
                x = shoulder.x + displacement_x
                shoulder.z = image_smoothing(x, shoulder.y, depth_image, depth_scale)
                if shoulder.z > 1.26 * median_z or shoulder.z == 0 or shoulder.z < median_z:
                    displacement_x += 10
                else:
                    flag = False
                if x > shoulder_lim.x:
                    break
    return shoulder.z, flag

# Correction of abnormal values due to scattering
def correct_values(bookmarks, depth_image, depth_scale):
    """Corrects the depth values of the bookmarks."""
    if bookmarks:
        print("The number of values sent is verified")
        # Face markers
        nose = bookmarks[0]
        eye_right = bookmarks[7]
        eye_left = bookmarks[8]
        mouth_right = bookmarks[9]
        mouth_left = bookmarks[10]
        # Right arm markers
        right_shoulder = bookmarks[11]
        right_elbow = bookmarks[13]
        right_wrist = bookmarks[15]
        # Left arm markers
        left_shoulder = bookmarks[12]
        left_elbow = bookmarks[14]
        left_wrist = bookmarks[16]
        # Hips markers
        right_hip = bookmarks[23]
        left_hip = bookmarks[24]
        # Calculate initial depth values
        nose.z = image_smoothing(nose.x, nose.y, depth_image, depth_scale)
        eye_right.z = image_smoothing(eye_right.x, eye_right.y, depth_image, depth_scale)
        eye_left.z = image_smoothing(eye_left.x, eye_left.y,depth_image, depth_scale)
        mouth_right.z = image_smoothing(mouth_right.x, mouth_right.y, depth_image, depth_scale)
        mouth_left.z = image_smoothing(mouth_left.x, mouth_left.y, depth_image, depth_scale)

        z_vals = [nose.z, eye_right.z, eye_left.z, mouth_right.z, mouth_left.z]
        median_z = round(np.median(z_vals), 2)
        nose.z = eye_right.z = eye_left.z = mouth_right.z = mouth_left.z = median_z
        print(f"Median Z : {median_z}")

        # Right shoulder correction
        right_shoulder.z, _ = shoulder_correction(right_shoulder, left_shoulder, median_z, depth_image, depth_scale)

        # Left shoulder correction
        left_shoulder.z, _ = shoulder_correction(left_shoulder, right_shoulder, median_z, depth_image, depth_scale)

       # Smoothing other markers
        right_elbow.z = image_smoothing(right_elbow.x, right_elbow.y, depth_image, depth_scale)
        right_wrist.z = image_smoothing(right_wrist.x, right_wrist.y, depth_image, depth_scale)
        right_hip.z = image_smoothing(right_hip.x, right_hip.y, depth_image, depth_scale)
        left_elbow.z = image_smoothing(left_elbow.x, left_elbow.y, depth_image, depth_scale)
        left_wrist.z = image_smoothing(left_wrist.x, left_wrist.y, depth_image, depth_scale)
        left_hip.z = image_smoothing(left_hip.x, left_hip.y, depth_image, depth_scale)

        angle = round(angle_arm(right_shoulder, right_elbow, right_wrist), 2)

        if 135 <= angle and not is_nearby(right_shoulder, right_wrist):
            print(f"{angle}° Right arm fully extended...")
            # Wrist correction
            if right_shoulder.z * 1.2 < right_wrist.z:
                print(f"Right wrist Z {right_wrist.z}")
                right_wrist.z = right_shoulder.z
                print(f"Right wrist Z correct {right_wrist.z}")
                if right_shoulder.z * 1.2 <= right_elbow.z:
                    print(f"Right elbow Z {right_elbow.z}")
                    right_elbow.z = right_shoulder.z
                    print(f"Right wrist Z correct {right_elbow.z}")
            elif right_wrist.z == 0:
                if right_elbow.z <= right_shoulder.z * 1.2:
                    print("Interpolated right wrist")
                    right_wrist.z = round((right_shoulder.z + right_elbow.z) / 2.0, 2)
            elif right_wrist.z <= right_shoulder.z * 1.2 and right_shoulder.z * 1.2 < right_elbow.z:
                print("Interpolated elbow")
                print(f"Elbow {right_elbow.z}")
                right_elbow.z = round((right_shoulder.z + right_wrist.z) / 2, 2)
                print(f"Elbow correct: {right_elbow.z}")
            if right_shoulder.z * 1.2 < right_wrist.z:
                right_wrist.z = right_shoulder.z
            if right_shoulder.z * 1.2 < right_elbow.z:
                right_elbow.z = right_shoulder.z

        elif 45 <= angle <135 and not is_nearby(right_shoulder, right_wrist):
            print(f"{angle}° Right arm moderately flexed...")

            if right_shoulder.x * 0.9 < right_elbow.x < right_shoulder.x * 1.2 and right_elbow.y < right_wrist.y:
                print("Overlaping elbow shoulder")
                right_shoulder.z = left_shoulder.z
                right_elbow.x = right_shoulder.x * 1.1
                right_elbow.y = right_shoulder.y
            else:
                if right_shoulder.y < right_wrist.y:
                    if right_wrist.z == 0:
                        right_wrist.z = right_shoulder.z
                    if right_shoulder.y * 0.4 <= right_elbow.y <= right_shoulder.y * 1.2:
                        print("SAY HELLO TO MY LITTLE FRIEND")
                        if right_shoulder.z < right_wrist.z:
                            right_wrist.z = right_shoulder.z
                        if right_shoulder.z * 1.2 < right_elbow.z:
                            #right_wrist.z = right_shoulder.z
                            if right_wrist.z < right_elbow.z * 1.2:
                                right_elbow.z = round((right_shoulder.z + right_wrist.z) / 2, 2)
                                right_wrist.z = right_shoulder.z
                if right_shoulder.y * 1.2 < right_elbow.y and right_shoulder.y * 1.2 < right_wrist.y:
                    print("Hand over the head")
                    if right_shoulder.z * 1.2 < right_wrist.z:
                        right_wrist.z = round((median_z + right_shoulder.z) / 2, 2)
                        if right_shoulder.z * 0.9 <= right_elbow.z <= right_shoulder.z * 1.2:
                            print("Ok elbow")
                        else:
                            right_elbow.z = right_wrist.z
                    if right_shoulder.z < right_elbow.z:
                        right_elbow.z = round((right_shoulder.z + right_wrist.z) / 2, 2)
                    else:
                        right_elbow.z = round((right_shoulder.z + right_wrist.z) / 2, 2)
                elif right_shoulder.y > right_wrist.y:
                    if right_shoulder.z * 1.2 < right_elbow.z:
                        right_elbow.z = right_shoulder.z
                    if is_nearby(right_wrist, right_hip):
                        if right_shoulder.z * 1.2 < right_elbow.z:
                            right_elbow.z = right_shoulder.z

        elif 20 < angle < 45 and not is_nearby(right_shoulder, right_elbow) and right_elbow.x > right_shoulder.x * 1.2:
            print(f"{angle}° Right arm flexed...")
            if right_shoulder.y * 0.7 <= right_wrist.y <= right_shoulder.y * 1.2:
                if right_shoulder.z * 0.7 <= right_wrist.z <= right_shoulder.z * 1.2:
                    if right_shoulder.z * 1.2 < right_elbow.z:
                        right_elbow.z = right_shoulder.z
                if right_shoulder.z < right_wrist.z and right_shoulder.z < right_elbow.z:
                    right_elbow.z = right_shoulder.z
                    right_wrist.z = right_shoulder.z
            else:
                right_elbow.z = right_shoulder.z
                right_wrist.z = right_shoulder.z

        if right_shoulder.y * 0.7 <= right_elbow.y <= right_shoulder.y * 0.9:
            print("Extencion FRONTAL")


        # # Correction of hips values
        if right_hip.z > right_shoulder.z or left_hip.z > left_hip.z:
             right_hip.z, left_hip.z = right_shoulder.z, left_shoulder.z

        bookmarks[0] = nose
        bookmarks[7] = eye_right
        bookmarks[8] = eye_left
        bookmarks[9] = mouth_right
        bookmarks[10] = mouth_left
        # Right arm markers
        bookmarks[11] = right_shoulder
        bookmarks[13] = right_elbow
        bookmarks[15] = right_wrist
        # Left arm markers
        bookmarks[12] = left_shoulder
        bookmarks[14] = left_elbow
        bookmarks[16] = left_wrist
        # Hips markers
        bookmarks[23] = right_hip
        bookmarks[24] = left_hip
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
            cv2.circle(depth_image, (x, STREAM_RES_Y - y), 5, (255, 0, 0), -1)
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
    buffer_bookmarks = []
    try:
        x_aux, y_aux, z_aux = None, None, None

        while True:
            color_image, depth_image = realsense_camera.getImageRGBDepth()
            store_bookmarks = []
            segmentation_mask = None
            if color_image is not None:
                pose_landmarks_list, _ = get_pose_landmarks(color_image)

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
                    draw_pose_markers_on_depth_image_from_bookmarks(depth_image, store_bookmarks)
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

                    bookmarks = correct_values(store_bookmarks, depth_image, depth_scale)
                    # print(f"Len of Bookmarks: {len(bookmarks)}")
                    sequence_number += 1
                    # Send UDP message with bookmarks
                    send_udp_message(sequence_number, [coord for bookmark in bookmarks for coord in
                                                       (bookmark.x, bookmark.y, bookmark.z)])
                    # Draw pose markers on depth image
                    draw_pose_markers_on_depth_image_from_bookmarks(color_image, bookmarks)

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