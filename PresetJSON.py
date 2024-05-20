import pyrealsense2 as rs
import time
import json

DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C", "0B5B"]

def find_device_that_supports_advanced_mode():
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
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

    #Guardar configuración en un archivo JSON
    with open('camera_config.json', 'w') as json_file:
        json_file.write(serialized_string)

    # Leer configuración desde el archivo JSON
    with open('presets/HighAccuracyPreset.json', 'r') as json_file:
        serialized_string = json_file.read()

    # Cargar configuración desde la cadena JSON
    advnc_mode.load_json(serialized_string)

except Exception as e:
    print(e)
    pass
