import pyrealsense2 as rs
import threading
import cv2
import time
import json
import numpy as np

class IntelRealSenseD435:
    def __init__(self, streamResX, streamResY, fps, presetJSON):
        self.streamResX = streamResX
        self.streamResY = streamResY
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.depthScale = None
        self.running = False
        self.frames = None
        self.mutex = threading.Lock()
        self.presetJSON = presetJSON

        # Filtros de profundidad
        self.depth_to_disparity = rs.disparity_transform(True)
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.disparity_to_depth = rs.disparity_transform(False)

    def configurePreset(self):
        DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03",
                           "0B07", "0B3A", "0B5C", "0B5B"]

        def find_device_that_supports_advanced_mode():
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                if dev.supports(rs.camera_info.product_id) and str(
                        dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
                    if dev.supports(rs.camera_info.name):
                        print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
                    return dev
            raise Exception("No D400 product line device that supports advanced mode was found")

        try:
            dev = find_device_that_supports_advanced_mode()
            advnc_mode = rs.rs400_advanced_mode(dev)
            print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

            while not advnc_mode.is_enabled():
                print("Trying to enable advanced mode...")
                advnc_mode.toggle_advanced_mode(True)
                print("Sleeping for 5 seconds...")
                time.sleep(5)
                dev = find_device_that_supports_advanced_mode()
                advnc_mode = rs.rs400_advanced_mode(dev)
                print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

            # Mostrar configuración actual en pantalla
            serialized_string = advnc_mode.serialize_json()
            print("Controls as JSON: \n", serialized_string)

            # Guardar configuración en un archivo JSON
            with open('camera_config.json', 'w') as json_file:
                json_file.write(serialized_string)

            # Leer configuración desde el archivo JSON
            with open(self.presetJSON, 'r') as json_file:
                serialized_string = json_file.read()

            # Cargar configuración desde la cadena JSON
            advnc_mode.load_json(serialized_string)

        except Exception as e:
            print(e)
            pass

    def configureCamera(self):
        try:
            #self.configurePreset()
            self.config.enable_stream(rs.stream.depth, self.streamResX, self.streamResY, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.streamResX, self.streamResY, rs.format.bgr8, self.fps)
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"Error configuring the camera: {e}")
            raise

    def getDepthScale(self):
        try:
            profile = self.pipeline.get_active_profile()
            depth_sensor = profile.get_device().first_depth_sensor()
            return depth_sensor.get_depth_scale()
        except Exception as e:
            print(f"Error getting depth scale: {e}")
            return None

    def startCapture(self):
        self.running = True
        threading.Thread(target=self.captureThread, daemon=True).start()

    def captureThread(self):
        alignTo = rs.stream.color
        align = rs.align(alignTo)
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames()
                alignedFrames = align.process(frames)

                with self.mutex:
                    self.frames = alignedFrames
            except Exception as e:
                print(f"Error during frame capture: {e}")

    def checkCameraConnection(self):
        try:
            realsense_ctx = rs.context()
            serial_number = []
            connectionCamFlag = realsense_ctx.query_devices()

            if len(connectionCamFlag) > 0:
                msjConnection = 'The RS camera is connected'
                connectionCam = True
            else:
                msjConnection = 'The RS camera is not connected'
                connectionCam = False
        except Exception as e:
            msjConnection = f'Error when connecting device: {e}'
            connectionCam = False

        if connectionCam:
            for i in range(len(realsense_ctx.devices)):
                detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
                serial_number.append(detected_camera)
            device = serial_number[0]
        else:
            device = None

        return connectionCam, msjConnection, device

    def getImageRGBDepth(self):
        with self.mutex:
            if self.frames is None:
                return None, None

            alignedDepthFrame = self.frames.get_depth_frame()
            colorFrame = self.frames.get_color_frame()

        if not alignedDepthFrame or not colorFrame:
            return None, None

        try:
            # Aplicar filtros de profundidad
            depth_frame = self.depth_to_disparity.process(alignedDepthFrame)
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.disparity_to_depth.process(depth_frame)

            depthImage = np.asanyarray(depth_frame.get_data())
            depthImageFlipped = cv2.flip(depthImage, 1)

            colorImage = np.asanyarray(colorFrame.get_data())
            colorImages = cv2.flip(colorImage, 1)
            colorImageRGB = cv2.cvtColor(colorImages, cv2.COLOR_BGR2RGB)

            return colorImageRGB, depthImageFlipped
        except Exception as e:
            print(f"Error processing images: {e}")
            return None, None

    def stopCapture(self):
        self.running = False
        self.pipeline.stop()
