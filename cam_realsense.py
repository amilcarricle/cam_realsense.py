import json
import threading

import pyrealsense2 as rs
import numpy as np
import cv2

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

    def configureCamera(self):

        #preset = json.loads(preset_json)
        self.config.enable_stream(rs.stream.depth, self.streamResX, self.streamResY, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.streamResX, self.streamResY, rs.format.bgr8, self.fps)
        self.pipeline.start(self.config)
        self.configurePreset('high_accuracy')
        self.depthScale = self.getDepthScale()

    def configurePreset(self, preset):

        profile = self.pipeline.get_active_profile()
        depthSensor = profile.get_device().first_depth_sensor()
        self.setSensorOptions(depthSensor, preset)

    def setSensorOptions(self, sensor, preset):

        if preset == 'high_accuracy':
            sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
        elif preset == 'high_density':
            sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_density)

    def getDepthScale(self):

        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()

        return depth_sensor.get_depth_scale()

    def startCapture(self):
        self.running = True
        threading.Thread(target= self.captureThread, daemon= True).start()

    def captureThread(self):
        alignTo = rs.stream.color
        align = rs.align(alignTo)
        while self.running:
            frames = self.pipeline.wait_for_frames()
            alignedFrames = align.process(frames)

            with self.mutex:
                self.frames = alignedFrames

    def checkCameraConnection(self):

        realsense_ctx = rs.context()
        serial_number = []
        device = 0
        # Serves the connection  of a device
        try:
            connectionCamFlag = realsense_ctx.query_devices()

            if len(connectionCamFlag) > 0:
                msjConnection = 'The RS camera is connected'
                connectionCam = True
            else:
                msjConnection = 'The RS camera is not connected'
                connectionCam = False
        except Exception as exception:
            msjConnection = 'Error when connecting device'
            connectionCam = False

        # Getting device ID
        if connectionCam == True:
            for i in range(len(realsense_ctx.devices)):
                detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
                serial_number.append(detected_camera)
            device = serial_number[0]

        return connectionCam, msjConnection, device

    def getImageRGBDepth(self):

        with self.mutex:
            if self.frames is None:
                return None, None

            alignedDepthFrame = self.frames.get_depth_frame()
            colorFrame = self.frames.get_color_frame()

        if not alignedDepthFrame or not colorFrame:
            return None, None
        # Process image
        depthImage = np.asanyarray(alignedDepthFrame.get_data())
        depthImage = cv2.applyColorMap(cv2.convertScaleAbs(depthImage, alpha=0.03), cv2.COLORMAP_JET)
        depthImageFlipped = cv2.flip(depthImage, 1)
        colorImage = np.asanyarray(colorFrame.get_data())
        colorImages = cv2.flip(colorImage, 1)
        colorImageRGB = cv2.cvtColor(colorImages, cv2.COLOR_BGR2RGBA)

        return colorImageRGB, depthImageFlipped

    def stopCapture(self):
        self.running = False
        self.pipeline.stop()