import hashlib
import math

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
DIST_SHOULDER_ELBOW = 0.35
DIST_ELBOW_WRIST = 0.30
ARM_LENGHT = DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW
# Camera configuration
STREAM_RES_X = 640
STREAM_RES_Y = 480
FPS = 30
PRESET_JSON = 'BodyScanPreset.json'

# Segmentation mask
BG_COLOR = (125, 125, 125)
MASK_COLOR = (255, 255, 255)

# UDP server configuration
SERVER_ADDRESS = ("127.0.0.1", 5052)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# MediaPipe Pose configuration
# Model Path: Model selection (Lite, Full, Heavy)
# POSE_LANDMARKER_MODEL_PATH = "C:/Users/amilc/OneDrive/Escritorio/REpo/Tasks/pose_landmarker_lite.task"
POSE_LANDMARKER_MODEL_PATH = "pose_landmarker_full.task"
#POSE_LANDMARKER_MODEL_PATH = "pose_landmarker_heavy.task"

# Create a PoseLandmarker object
"""
ATTENTION!!!!!
Aclaracion, esta forma de crear los objetos difiere de los ejemplos mostrados por parte de google. Buscar en gitHub
un ejemplo donde se explica este cambio
"""
base_options = python.BaseOptions(model_asset_buffer=open(POSE_LANDMARKER_MODEL_PATH, "rb").read())
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# The number of columns and rows for the grid are defined
NUMBER_COLUMS = 8
NUMBER_ROWS = 6

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
    buffer_size = 50
    depth_values = depth_image[max(0, point_y - buffer_size):
                               min(STREAM_RES_Y, point_y + buffer_size + 1),
                               max(0, point_x - buffer_size):
                               min(STREAM_RES_X, point_x + buffer_size + 1)]
    z_values = depth_values.flatten() * depth_scale
    return round(np.median(z_values), 2)

def calculate_distance(marker1, marker2):
    """Calculates the Euclidean distance between two points."""
    return round(np.sqrt((marker2.x - marker1.x) ** 2 + (marker2.y - marker1.y) ** 2))

def is_nearby(marker1, marker2, threshold):
    """Checks if two markers are within a certain distance threshold."""
    distance = np.sqrt((marker1.x - marker2.x) ** 2 + (marker1.y - marker2.y) ** 2)
    return distance < threshold

def angle(shoulder, elbow, wrist):
    p1 = np.array([shoulder.x, shoulder.y])
    p2 = np.array([elbow.x, elbow.y])
    p3 = np.array([wrist.x, wrist.y])

    l1 = np.linalg.norm(p2 - p3)
    l2 = np.linalg.norm(p1 - p3)
    l3 = np.linalg.norm(p1 - p2)
    theta = (l1 ** 2 + l3 ** 2 - l2 ** 2) / (2 * l1 * l3)
    if -1 <= theta <= 1:
        return (degrees ( acos( theta ) ) )
    else:
        return 1

#Shoulder correction
def shoulder_correction(shoulder, shoulder_lim, median_z):
    flag = True
    if shoulder.z > 1.26 * median_z or shoulder.z == 0 or shoulder.z < median_z:
        displacement_x = 0
        #For right shoulder
        if shoulder_lim.x < shoulder.x:
            while flag and displacement_x <= STREAM_RES_X:
                x = shoulder.x - displacement_x
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
                if shoulder.z > 1.26 * median_z or shoulder.z == 0 or shoulder.z < median_z:
                    displacement_x += 10
                else:
                    flag = False
                if x > shoulder_lim.x:
                    break
    return shoulder.z, flag

# Correction of abnormal values due to scattering
def calculate_z(marker1, marker2, dist_right_shoulder_hip, depth_scale):
    """

    :param right_shoulder:
    :param right_elbow:
    :param dist_right_shoulder_hip:
    :param depth_scale:
    :return:
    """
    a = ((marker1.x - marker2.x) ** 2) * depth_scale
    a = round(a * (DIST_SHOULDER_ELBOW + DIST_ELBOW_WRIST) / dist_right_shoulder_hip, 2)
    b = ((marker1.y - marker2.y) ** 2) * depth_scale
    b = round(b * (DIST_SHOULDER_ELBOW + DIST_ELBOW_WRIST) / dist_right_shoulder_hip, 2)
    c = (DIST_SHOULDER_ELBOW ** 2) - a - b
    if c < 0:
        return 0
    else:
        return round(marker1.z - np.sqrt(c), 2)

def from_pixels_to_meters(pix_distance, dist_right_shoulder_hip):
    """
    :param pix_distance:
    :param dist_right_shoulder_hip:
    :return:
    """
    return round((pix_distance * (DIST_SHOULDER_ELBOW + DIST_ELBOW_WRIST) / dist_right_shoulder_hip) , 2)

def correct_values(bookmarks, buffer_bookmarks, sequence_number, depth_image, depth_scale):
    """Corrects the depth values of the bookmarks."""

    # print(f"Bookmarks: {bookmarks}")
    # print(f"Buffer: {buffer_bookmarks}")
    if bookmarks:

        if sequence_number == 1:
            buffer_bookmarks = bookmarks
            print("Inicial")
        else:
            print("Final")
            for i in range(len(bookmarks)):
                """
                Se busca corregir el efecto de oscilacion en las mediciones de profundidad, todos los valores 
                menores a un 5% entre la medicion anterior y la reciente seran descartados. Se comparan 
                valores en +/- 5 pixeles de x e y.
                """
                if abs(buffer_bookmarks[i].z - bookmarks[i].z) <= 0.05:
                    if (abs(buffer_bookmarks[i].x - bookmarks[i].x) <= 5
                            or abs(buffer_bookmarks[i].y - bookmarks[i].y) <= 5):
                        bookmarks[i].z = buffer_bookmarks[i].z
                        """
                        Mediante esta linea omito las oscilaciones de las coordenadas X e Y
                        """
                    if abs(buffer_bookmarks[i].x - bookmarks[i].x) <= 15:
                        bookmarks[i].x = buffer_bookmarks[i].x
                    if abs(buffer_bookmarks[i].y - bookmarks[i].y) <= 15:
                        bookmarks[i].y = buffer_bookmarks[i].y

        print("The number of values sent is verified")
        msj = "Initial msj"
        # Face markers
        nose = bookmarks[0]
        eye_right = bookmarks[1]
        eye_left = bookmarks[2]
        mouth_right = bookmarks[3]
        mouth_left = bookmarks[4]
        # Right arm markers
        right_shoulder = bookmarks[5]
        right_elbow = bookmarks[7]
        right_wrist = bookmarks[9]
        # Left arm markers
        left_shoulder = bookmarks[6]
        left_elbow = bookmarks[8]
        left_wrist = bookmarks[10]
        # Hips markers
        right_hip = bookmarks[11]
        left_hip = bookmarks[12]

        # Calculate initial depth values
        z_vals = [nose.z, eye_right.z, eye_left.z, mouth_right.z, mouth_left.z]
        median_z = round(np.median(z_vals), 2)
        nose.z = eye_right.z = eye_left.z = mouth_right.z = mouth_left.z = median_z
        print(f"Median Z : {median_z}")

        # Right shoulder correction

        right_shoulder.z, _ = shoulder_correction(right_shoulder, left_shoulder, median_z)
        # Left shoulder correction

        left_shoulder.z, _ = shoulder_correction(left_shoulder, right_shoulder, median_z)

        # Correction of hips values
        if right_hip.z > right_shoulder.z or left_hip.z > left_shoulder.z:
            right_hip.z, left_hip.z = right_shoulder.z, left_shoulder.z

        right_arm_angle = round(angle(right_shoulder, right_elbow, right_wrist), 2)
        right_trunk_angle = round(angle(right_hip, right_shoulder, right_elbow), 2)
        dist_right_shoulder_hip = calculate_distance(right_shoulder, right_hip)
        right_forearm = calculate_distance(right_shoulder, right_elbow)
        right_upperarm = calculate_distance(right_elbow, right_wrist)

        # We convert these ARM segments to meters
        right_forearm = from_pixels_to_meters(right_forearm, dist_right_shoulder_hip)
        right_upperarm = from_pixels_to_meters(right_upperarm, dist_right_shoulder_hip)

        """
        En las siguenientes lineas se busca corregir los valores pertenecientes a los marcadores de codo y munieca. 
        Por lo tanto, se busca ir corrigiendo todos los errores en cada pocicion de los brazos.
        Partiremos de una extencion conmpleta del brazo, en las mismas los marcadores estan formando un angulo mayor a 
        150°. Esto se cumole para las estencion completa lateral y la frontal parcial.
        """
        if right_arm_angle >= 150:
            #Extension completa lateral
            if (DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW) * 0.8 <= right_forearm + right_upperarm <= (
                    DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW) * 1.2:
                if right_elbow.z > right_shoulder.z:
                    right_elbow.z = right_shoulder.z
                if right_wrist.z > right_shoulder.z:
                    right_wrist.z = right_shoulder.z
            #Extension completa intermedia
            else:
                right_elbow.z = calculate_z(right_shoulder, right_elbow, dist_right_shoulder_hip, depth_scale)
                right_wrist.z = calculate_z(right_elbow, right_wrist, dist_right_shoulder_hip, depth_scale)
        """
        Seguimos con la extencion frontal total, aca surge un problema que es el que el marcador del codo se mueve y ya
        no se tiene un angulo de 150 o superior. Por lo tanto trabajaremos con la deteccion de proximidad de los 
        marcadores
        """
        if is_nearby(right_shoulder, right_wrist, 50) or is_nearby(right_shoulder, right_elbow, 50):
            if right_wrist.z >= right_shoulder.z:
                right_wrist.z = round(right_shoulder.z - (DIST_SHOULDER_ELBOW + DIST_ELBOW_WRIST), 2)
            if right_elbow.z >= right_shoulder.z:
                right_elbow.z = round(right_shoulder.z - DIST_SHOULDER_ELBOW, 2)
                if right_elbow.z < right_wrist.z:
                    right_elbow.z = round((right_shoulder.z + right_wrist.z) / 2, 2)
            msj = f"{right_elbow.z} // {right_wrist.z}"

        """
        Ahora vamos a la etapa donde el brazo forma un angulo menor a 90°, aqui puedo utilizar tambien el angulo de tronco
        """
        # Gesto de saludo. Extencion lateral
        if right_arm_angle < 150 and right_trunk_angle > 60:
            if (DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW) * 0.8 <= right_forearm + right_upperarm <= (
                    DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW) * 1.2:
                if right_elbow.z > right_shoulder.z:
                    right_elbow.z = right_shoulder.z
                if right_wrist.z > right_shoulder.z:
                    right_wrist.z = right_shoulder.z
            """
            Similar al saludo pero acercando el antebrazo
            """
            if right_upperarm < 0.85*DIST_ELBOW_WRIST and not is_nearby(right_wrist, eye_right, 20):
                right_elbow.z = right_shoulder.z
                right_wrist.z = calculate_z(right_elbow, right_wrist, dist_right_shoulder_hip, depth_scale)
                msj = "OK"
            """
            Si la munieca toca la cabeza y el codo esta flexionada hacia la camra
            """
            if is_nearby(right_wrist, eye_right, 20) and is_nearby(right_shoulder, right_elbow, 40):
                right_wrist.z = median_z
                right_elbow.z = calculate_z(right_shoulder, right_elbow, dist_right_shoulder_hip, depth_scale)
                msj = f"{right_elbow.z} // {right_wrist.z}"

        """
        Para la parte inferior, es decir los movimientos por debajo de un angulo de tronco menor a 90 °
        """
        """
        La extension completa lateral ya esta funcionado en base a las lineas de codigo en la parte superior
        """
        """
        Para el gesto de saludo, pero para la parte inferior
        """
        if right_arm_angle < 150 and right_trunk_angle < 90:
            if (DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW) * 0.8 <= right_forearm + right_upperarm <= (
                    DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW) * 1.2:
                if right_elbow.z > right_shoulder.z:
                    right_elbow.z = right_shoulder.z
                if right_wrist.z > right_shoulder.z:
                    right_wrist.z = right_shoulder.z
            if right_upperarm < 0.85*DIST_ELBOW_WRIST and not is_nearby(right_wrist, eye_right, 20):
                right_elbow.z = right_shoulder.z
                right_wrist.z = calculate_z(right_elbow, right_wrist, dist_right_shoulder_hip, depth_scale)
                msj = "OK"



        if right_elbow.z > right_shoulder.z:
            right_elbow.z = calculate_z(right_shoulder, right_elbow, dist_right_shoulder_hip, depth_scale)
        if right_wrist.z > right_shoulder.z:
            right_wrist.z = calculate_z(right_elbow, right_wrist, dist_right_shoulder_hip, depth_scale)

        left_arm_angle = round(angle(left_shoulder, left_elbow, left_wrist), 2)
        left_trunk_angle = round(angle(left_hip, left_shoulder, left_elbow), 2)
        dist_left_shoulder_hip = calculate_distance(left_shoulder, left_hip)
        left_forearm = calculate_distance(left_shoulder, left_elbow)
        left_upperarm = calculate_distance(left_elbow, left_wrist)

        # We convert these ARM segments to meters
        left_forearm = from_pixels_to_meters(left_forearm, dist_left_shoulder_hip)
        left_upperarm = from_pixels_to_meters(left_upperarm, dist_left_shoulder_hip)
        """
        En las siguenientes lineas se busca corregir los valores pertenecientes a los marcadores de codo y munieca. 
        Por lo tanto, se busca ir corrigiendo todos los errores en cada pocicion de los brazos.
        Partiremos de una extencion conmpleta del brazo, en las mismas los marcadores estan formando un angulo mayor a 
        150°. Esto se cumole para las estencion completa lateral y la frontal parcial.
        """
        if left_arm_angle >= 150:

            # Extension completa lateral
            if (DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW) * 0.8 <= left_forearm + left_upperarm <= (
                    DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW) * 1.2:
                if left_elbow.z > left_shoulder.z:
                    left_elbow.z = left_shoulder.z
                if left_wrist.z > left_shoulder.z:
                    left_wrist.z = left_shoulder.z
                msj = "HOLA"
            # Extension completa intermedia
            else:
                left_elbow.z = calculate_z(left_shoulder, left_elbow, dist_left_shoulder_hip, depth_scale)
                left_wrist.z = calculate_z(left_elbow, left_wrist, dist_left_shoulder_hip, depth_scale)


        if is_nearby(left_shoulder, left_wrist, 50) or is_nearby(left_shoulder, left_elbow, 50):
            if left_wrist.z >= left_shoulder.z:
                left_wrist.z = round(left_shoulder.z - (DIST_SHOULDER_ELBOW + DIST_ELBOW_WRIST), 2)
            if left_elbow.z >= left_shoulder.z:
                left_elbow.z = round(left_shoulder.z - DIST_SHOULDER_ELBOW, 2)
                if left_elbow.z < left_wrist.z:
                    left_elbow.z = round((left_shoulder.z + left_wrist.z) / 2, 2)
            msj = f"{left_elbow.z} // {left_wrist.z}"

        # Gesto de saludo. Extencion lateral
        if left_arm_angle < 150 and left_trunk_angle > 60:
            if (DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW) * 0.8 <= left_forearm + left_upperarm <= (
                    DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW) * 1.2:
                if left_elbow.z > left_shoulder.z:
                    left_elbow.z = left_shoulder.z
                if left_wrist.z > left_shoulder.z:
                    left_wrist.z = left_shoulder.z
    
            if left_upperarm < 0.85 * DIST_ELBOW_WRIST and not is_nearby(left_wrist, eye_left, 20):
                left_elbow.z = left_shoulder.z
                left_wrist.z = calculate_z(left_elbow, left_wrist, dist_left_shoulder_hip, depth_scale)
                msj = "OK"
          
            if is_nearby(left_wrist, eye_left, 20) and is_nearby(left_shoulder, left_elbow, 40):
                left_wrist.z = median_z
                left_elbow.z = calculate_z(left_shoulder, left_elbow, dist_left_shoulder_hip, depth_scale)
                msj = f"{left_elbow.z} // {left_wrist.z}"

       
        if left_arm_angle < 150 and left_trunk_angle < 90:
            if (DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW) * 0.8 <= left_forearm + left_upperarm <= (
                    DIST_ELBOW_WRIST + DIST_SHOULDER_ELBOW) * 1.2:
                if left_elbow.z > left_shoulder.z:
                    left_elbow.z = left_shoulder.z
                if left_wrist.z > left_shoulder.z:
                    left_wrist.z = left_shoulder.z
            if left_upperarm < 0.85 * DIST_ELBOW_WRIST and not is_nearby(left_wrist, eye_left, 20):
                left_elbow.z = left_shoulder.z
                left_wrist.z = calculate_z(left_elbow, left_wrist, dist_left_shoulder_hip, depth_scale)
                msj = "OK"

        if left_elbow.z > left_shoulder.z:
            left_elbow.z = calculate_z(left_shoulder, left_elbow, dist_left_shoulder_hip, depth_scale)
        if left_wrist.z > left_shoulder.z:
            left_wrist.z = calculate_z(left_elbow, left_wrist, dist_left_shoulder_hip, depth_scale)

        bookmarks[0] = nose
        bookmarks[1] = eye_right
        bookmarks[2] = eye_left
        bookmarks[3] = mouth_right
        bookmarks[4] = mouth_left
        # Right arm markers
        bookmarks[5] = right_shoulder
        bookmarks[7] = right_elbow
        bookmarks[9] = right_wrist
        # Left arm markers
        bookmarks[6] = left_shoulder
        bookmarks[8] = left_elbow
        bookmarks[10] = left_wrist
        # Hips markers
        bookmarks[11] = right_hip
        bookmarks[12] = left_hip
        
        for i in range(len(bookmarks)):
            """
            Se busca corregir el efecto de oscilacion en las mediciones de profundidad, todos los valores 
            menores a un 5% entre la medicion anterior y la reciente seran descartados. Se comparan 
            valores en +/- 5 pixeles de x e y.
            """
            if abs(buffer_bookmarks[i].z - bookmarks[i].z) <= 0.05:
                if (abs(buffer_bookmarks[i].x - bookmarks[i].x) <= 5
                        or abs(buffer_bookmarks[i].y - bookmarks[i].y) <= 5):
                    bookmarks[i].z = buffer_bookmarks[i].z
                    """
                    Mediante esta linea omito las oscilaciones de las coordenadas X e Y
                    """
                if abs(buffer_bookmarks[i].x - bookmarks[i].x) <= 15:
                    bookmarks[i].x = buffer_bookmarks[i].x
                if abs(buffer_bookmarks[i].y - bookmarks[i].y) <= 15:
                    bookmarks[i].y = buffer_bookmarks[i].y

    else:
        print("The number of values sent is incorrect")

    return bookmarks, msj


def draw_pose_markers_on_depth_image_from_bookmarks(depth_image, bookmarks):
    """Draws pose markers on the depth image using a list of bookmarks."""
    for bookmark in bookmarks:
        x, y, z = bookmark.x, bookmark.y, bookmark.z
        if 0 <= y < STREAM_RES_Y and 0 <= x < STREAM_RES_X:
            x = int(x)
            y = int(y)
            cv2.circle(depth_image, (x, STREAM_RES_Y - y), 5, (0, 255, 0), -1)
            cv2.putText(depth_image, f'{z:.2f}', (x, STREAM_RES_Y - y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_lines(image):
    cell_height = STREAM_RES_Y // NUMBER_ROWS
    cell_width = STREAM_RES_X // NUMBER_COLUMS
    # Dibujar las líneas horizontales
    for i in range(1, NUMBER_ROWS):
        start_point = (0, i * cell_height)
        end_point = (STREAM_RES_X, i * cell_height)
        cv2.line(image, start_point, end_point, (0, 255, 0), 1)  # Color verde, grosor 1

    # Dibujar las líneas verticales
    for j in range(1, NUMBER_COLUMS):
        start_point = (j * cell_width, 0)
        end_point = (j * cell_width, STREAM_RES_Y)
        cv2.line(image, start_point, end_point, (0, 255, 0), 1)  # Color verde, grosor 1

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
                                z = round(depth_image[y, x] * depth_scale, 2)
                                x_aux, y_aux, z_aux = x, y, z
                            else:
                                x, y, z = x_aux, y_aux, z_aux
                            store_bookmarks.append(Bookmark(x, y, z))

                            # Print the coordinates
                            #print(f"Landmark {idx}: (x={x}, y={y}, z={z})")
                    aux_list = [store_bookmarks[0], store_bookmarks[7], store_bookmarks[8], store_bookmarks[9],
                                store_bookmarks[10], store_bookmarks[11], store_bookmarks[12], store_bookmarks[13],
                                store_bookmarks[14], store_bookmarks[15], store_bookmarks[16], store_bookmarks[23],
                                store_bookmarks[24]]
                    draw_pose_markers_on_depth_image_from_bookmarks(depth_image, aux_list)
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

                    # Smoothing markers
                    for i in range(len(aux_list)):

                        aux_list[i].z = image_smoothing(aux_list[i].x, aux_list[i].y, depth_image, depth_scale)


                    bookmarks, msj = correct_values(aux_list, buffer_bookmarks, sequence_number,
                                                    depth_image, depth_scale)


                    sequence_number += 1
                    buffer_bookmarks = bookmarks


                    # Send UDP message with bookmarks
                    send_udp_message(sequence_number, [coord for bookmark in bookmarks for coord in
                                                       (bookmark.x, bookmark.y, bookmark.z)])

                    # Text in color image
                    cv2.putText(color_image, msj, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    #msj_dist = f"(H-C: {forearm}); (C-M: {upper_arm}); (H-M: {arm})"
                    #cv2.putText(color_image, msj_dist, (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    # r_up_fo = round(upper_arm / forearm, 2)
                    # r_fo_a = round(forearm / arm, 2)
                    # r_up_a = round(upper_arm / arm, 2)
                    # msj_rel = f"(UP//FA: {r_up_fo}); (FA//A: {r_fo_a}); (UP//A: {r_up_a})"
                    # cv2.putText(color_image, msj_rel, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (125, 0, 125), 2, cv2.LINE_AA)

                    # Draw lines on color image
                    #draw_lines(color_image)

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