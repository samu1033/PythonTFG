import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

print("Visualizando camara... Pulsa 'q' para salir.")

try:
    while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())

        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16, valores en mm
        depth_gray = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_img = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)  # 3 canales para el hstack

        # Mostrar ambas imagenes una al lado de la otra
        combined = np.hstack((color_img, depth_img))
        cv2.imshow("RealSense  |  Color (izq)  -  Profundidad (der)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camara cerrada.")
