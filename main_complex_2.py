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
FPS = 15
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
POSE_LANDMARKER_MODEL_PATH = "C:/Users/Ricle/Desktop/Amilcar/CamRSManager/model packages/pose_landmarker_full.task"
#POSE_LANDMARKER_MODEL_PATH = "C:/Users/Ricle/Desktop/Amilcar/CamRSManager/model packages/pose_landmarker_heavy.task"

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
def is_full_extension(p1, p2, p3, tolerance=5):
    """Check if three points are aligned within a certain tolerance."""
    return abs((p3.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p3.x - p1.x)) < tolerance

# Function to check if depth values are too close (indicating a flexion)
def is_flexion(z_shoulder, z_elbow, z_wrist, threshold=0.1):
    return abs(z_shoulder - z_elbow) < threshold and abs(z_elbow - z_wrist) < threshold

# Function to check if the markers are close in the XY plane
def is_markers_close(p1, p2, p3, distance_threshold=30):
    distance_shoulder_elbow = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    distance_elbow_wrist = np.sqrt((p2.x - p3.x) ** 2 + (p2.y - p3.y) ** 2)
    return distance_shoulder_elbow < distance_threshold and distance_elbow_wrist < distance_threshold

def correct_values(bookmarks, depth_image, depth_scale):
    """Corrects the depth values of the bookmarks."""
    if bookmarks:
        print("The number of values sent is verified")
        # Face markers
        nose = bookmarks[0]
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
        mouth_right.z = image_smoothing(mouth_right.x, mouth_right.y, depth_image, depth_scale)
        mouth_left.z = image_smoothing(mouth_left.x, mouth_left.y, depth_image, depth_scale)

        z_vals = [nose.z, mouth_right.z, mouth_left.z]
        median_z = round(np.median(z_vals), 2)
        print(f"Median Z : {median_z}")

        # Right shoulder correction
        if right_shoulder.z > 1.26 * median_z or right_shoulder.z == 0 or right_shoulder.z < median_z:
            displacement_x = 0
            flag = True
            while flag and displacement_x <= STREAM_RES_X:
                x = right_shoulder.x - displacement_x
                right_shoulder.z = image_smoothing(x, right_shoulder.y, depth_image, depth_scale)
                if right_shoulder.z > 1.26 * median_z or right_shoulder.z == 0 or right_shoulder.z < median_z:
                    displacement_x += 10
                else:
                    flag = False
                if x < left_shoulder.x:
                    break

        # Left shoulder correction
        if left_shoulder.z > 1.26 * median_z or left_shoulder.z == 0 or left_shoulder.z < median_z:
            displacement_x = 0
            flag = True
            while flag and displacement_x <= STREAM_RES_X:
                x = left_shoulder.x + displacement_x
                left_shoulder.z = image_smoothing(x, left_shoulder.y, depth_image, depth_scale)
                if left_shoulder.z > 1.26 * median_z or left_shoulder.z == 0 or left_shoulder.z < median_z:
                    displacement_x += 10
                else:
                    flag = False
                if x > right_shoulder.x:
                    break

        if right_wrist.z > right_shoulder.z:
            if is_nearby(right_shoulder, right_hip):
                print("Right wrist is near the right hip, correcting depth value...")
                right_wrist.z = right_shoulder.z
                print(f"Assigned wrist Z value from shoulder: {right_shoulder.z}")
            else:
                print("Correcting right wrist depth value due to scattering...")
                right_wrist.z = right_shoulder.z
                print(f"Assigned wrist Z value from shoulder: {right_shoulder.z}")
        else:
            print("Wrist depth value within acceptable range.")

        # Right elbow correction
        if right_elbow.z == 0:
            print("Correcting right elbow depth value due to scattering...")
            x, y = right_elbow.x, right_elbow.y
            # Apply median filter to smooth depth value around the elbow
            elbow_depth_values = []
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < STREAM_RES_X and 0 <= ny < STREAM_RES_Y:
                        elbow_depth_values.append(depth_image[ny, nx] * depth_scale)
            if elbow_depth_values:
                filtered_z = round(np.median(elbow_depth_values), 2)
                right_elbow.z = filtered_z
                print(f"Filtered Z value for right elbow: {filtered_z}")
            # Interpolation using shoulder and wrist
            interpolated_z = (right_shoulder.z + right_wrist.z) / 2.0
            if abs(interpolated_z - right_elbow.z) > 0.1:  # Threshold for considering significant scattering
                right_elbow.z = round(interpolated_z, 2)
                print(f"Interpolated Z value for right elbow: {interpolated_z}")

        if right_elbow.z >= right_shoulder.z and right_elbow.x <= STREAM_RES_X:
            print("Correcting right elbow depth value due to scattering...")
            x, y = right_elbow.x, right_elbow.y
            # Apply median filter to smooth depth value around the elbow
            elbow_depth_values = []
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < STREAM_RES_X and 0 <= ny < STREAM_RES_Y:
                        elbow_depth_values.append(depth_image[ny, nx] * depth_scale)
            if elbow_depth_values:
                filtered_z = round(np.median(elbow_depth_values), 2)
                right_elbow.z = filtered_z
                print(f"Filtered Z value for right elbow: {filtered_z}")
            # Interpolation using shoulder and wrist
            interpolated_z = (right_shoulder.z + right_wrist.z) / 2.0
            if abs(interpolated_z - right_elbow.z) > 0.1:  # Threshold for considering significant scattering
                right_elbow.z = round(interpolated_z, 2)
                print(f"Interpolated Z value for right elbow: {interpolated_z}")
        else:
            print("Elbow depth value within acceptable range.")

        # # Check if right shoulder and elbow are very close
        # if is_nearby(right_shoulder, right_elbow):
        #     if is_nearby(right_elbow, right_wrist):
        #         print("Right shoulder, elbow, and wrist are very close, estimating Z values...")
        #         left_shoulder_elbow_dist = abs(left_shoulder.z - left_elbow.z)
        #         left_elbow_wrist_dist = abs(left_elbow.z - left_wrist.z)
        #
        #         # Estimating the Z values
        #         right_shoulder.z = left_shoulder.z
        #         right_elbow.z = right_shoulder.z + left_shoulder_elbow_dist
        #         right_wrist.z = right_elbow.z + left_elbow_wrist_dist
        #
        #         print(f"Estimated right shoulder Z: {right_shoulder.z}")
        #         print(f"Estimated right elbow Z: {right_elbow.z}")
        #         print(f"Estimated right wrist Z: {right_wrist.z}")

        # Left wrist correction
        if left_wrist.z > left_shoulder.z:
            if is_nearby(left_shoulder, left_hip):
                print("Left wrist is near the left hip, correcting depth value...")
                left_wrist.z = left_shoulder.z
                print(f"Assigned wrist Z value from shoulder: {left_shoulder.z}")
            else:
                print("Correcting left wrist depth value due to scattering...")
                left_wrist.z = left_shoulder.z
                print(f"Assigned wrist Z value from shoulder: {left_shoulder.z}")
        else:
            print("Wrist depth value within acceptable range.")

        # Left elbow correction
        if left_elbow.z == 0:
            print("Correcting left elbow depth value due to scattering...")
            x, y = left_elbow.x, left_elbow.y
            # Apply median filter to smooth depth value around the elbow
            elbow_depth_values = []
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < STREAM_RES_X and 0 <= ny < STREAM_RES_Y:
                        elbow_depth_values.append(depth_image[ny, nx] * depth_scale)
            if elbow_depth_values:
                filtered_z = round(np.median(elbow_depth_values), 2)
                left_elbow.z = filtered_z
                print(f"Filtered Z value for left elbow: {filtered_z}")
            # Interpolation using shoulder and wrist
            interpolated_z = (left_shoulder.z + left_wrist.z) / 2.0
            if abs(interpolated_z - left_elbow.z) > 0.1:  # Threshold for considering significant scattering
                left_elbow.z = round(interpolated_z, 2)
                print(f"Interpolated Z value for left elbow: {interpolated_z}")

        if left_elbow.z >= left_shoulder.z and left_elbow.x <= STREAM_RES_X:
            print("Correcting left elbow depth value due to scattering...")
            x, y = left_elbow.x, left_elbow.y
            # Apply median filter to smooth depth value around the elbow
            elbow_depth_values = []
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < STREAM_RES_X and 0 <= ny < STREAM_RES_Y:
                        elbow_depth_values.append(depth_image[ny, nx] * depth_scale)
            if elbow_depth_values:
                filtered_z = round(np.median(elbow_depth_values), 2)
                left_elbow.z = filtered_z
                print(f"Filtered Z value for left elbow: {filtered_z}")
            # Interpolation using shoulder and wrist
            interpolated_z = (left_shoulder.z + left_wrist.z) / 2.0
            if abs(interpolated_z - left_elbow.z) > 0.1:  # Threshold for considering significant scattering
                left_elbow.z = round(interpolated_z, 2)
                print(f"Interpolated Z value for left elbow: {interpolated_z}")
        else:
            print("Elbow depth value within acceptable range.")

        # Check if right shoulder and elbow are very close
        if is_nearby(right_shoulder, right_elbow) and not is_nearby(right_elbow, right_wrist):
            print("Right arm appears to be fully extended, estimating Z values based on left arm distances...")
            right_shoulder.z = left_shoulder.z
            right_elbow.z = left_shoulder.z + (left_elbow.z - left_shoulder.z)
            right_wrist.z = right_elbow.z + (left_wrist.z - left_elbow.z)
            print(f"Estimated right shoulder Z: {right_shoulder.z}")
            print(f"Estimated right elbow Z: {right_elbow.z}")
            print(f"Estimated right wrist Z: {right_wrist.z}")
        if is_nearby(right_shoulder, right_elbow) and is_nearby(right_elbow, right_wrist):
            print("Right shoulder, elbow, and wrist are very close, estimating Z values...")
            left_shoulder_elbow_dist = abs(left_shoulder.z - left_elbow.z)
            left_elbow_wrist_dist = abs(left_elbow.z - left_wrist.z)
            # Estimating the Z values
            right_shoulder.z = left_shoulder.z
            right_elbow.z = right_shoulder.z + left_shoulder_elbow_dist
            right_wrist.z = right_elbow.z + left_elbow_wrist_dist

            print(f"Estimated right shoulder Z: {right_shoulder.z}")
            print(f"Estimated right elbow Z: {right_elbow.z}")
            print(f"Estimated right wrist Z: {right_wrist.z}")

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

                    bookmarks = correct_values(store_bookmarks, depth_image, depth_scale)
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