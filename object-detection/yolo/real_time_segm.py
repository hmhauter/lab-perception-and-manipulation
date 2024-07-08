import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Test the model in real-time with the Intel RealSense Camera

# Confidence Threshold for Model
PLATE_THRESHOLD = 0.7

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# MODEL 
plate_model_segm = YOLO("object-detection/model/best.pt")

# Start the pipeline
pipeline.start(config)


try:
    while True:
        # Wait for the next set of frames
        frames = pipeline.wait_for_frames()

        # Get the color frame
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        
        if not color_frame:
            continue


        # Convert the color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        input_depth = depth_image.astype(np.float32)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        segmented_image = np.ones_like(color_image)
        prediction = plate_model_segm(color_image)
        num_predictions = len(prediction[0].boxes.conf) 
        if num_predictions > 0:
            # We predicted a plate in the scene 
            for indx in range(num_predictions):
                polygon_xy = prediction[0].masks[indx].xy[0]

                from matplotlib.patches import Polygon
                # Create a polygon patch
                polygon_patch = Polygon(polygon_xy, closed=True, edgecolor='r', linewidth=2, fill=False)

                # Create a binary mask from the polygon
                polygon_mask = np.zeros_like(color_image[:, :, 0], dtype=np.uint8)
                polygon_mask = cv2.fillPoly(polygon_mask, [polygon_xy.astype(np.int32)], color=255)

                # Apply the mask to the original image
                segmented_image = cv2.bitwise_and(color_image, color_image, mask=polygon_mask)

                # If the confidence is bigger than a certain threshold we are pretty sure we found a valid plate 
                if prediction[0].masks[indx].conf[0] > PLATE_THRESHOLD:
                    print("Valid Detection")


        # Display the original frame
        combined_image = np.hstack((color_image, segmented_image))
        cv2.imshow('Original Frame', combined_image)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
