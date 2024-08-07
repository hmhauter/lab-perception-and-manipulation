from ultralytics import YOLO
import numpy as np
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt
import cv2

"""
YOLO Model that returns bounding box, confidence and segmentation of a multi-well plate from a RGB image
"""

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, rgb_image):
        "Function to detect the plate in the image and return the bounding box."
        rgb_image_array = np.asarray(rgb_image)
        prediction = self.model(rgb_image_array)
        if len(prediction[0].boxes) == 0:
            print("No plate detected")
            return None
        else:
            print(prediction[0].boxes[0].xywh)
            return prediction[0].boxes[0].xywh

    def detect_and_conf(self, rgb_image):
        "Function to detect the plate in the image and return the bounding box."
        rgb_image_array = np.asarray(rgb_image)
        prediction = self.model(rgb_image_array)
        if len(prediction[0].boxes) == 0:
            print("No plate detected")
            return None
        else:
            print(prediction[0].boxes[0].xywh)
            return prediction[0].boxes[0].xywh, prediction[0].boxes[0].conf

    def segment_and_detect(self, rgb_image, depth_data, debug=False):
        "Function to segment the plate in the image and return the segmented image and depth data."
        segmented_image = None
        segmented_depth = None
        rgb_image_array = np.asarray(rgb_image)
        h, w = rgb_image_array.shape[:2]
        prediction = self.model(rgb_image_array)

        if prediction[0].masks != None:
            polygon_xy = prediction[0].masks[0].xy[0]

            polygon_patch = Polygon(
                polygon_xy, closed=True, edgecolor="r", linewidth=2, fill=False
            )

            polygon_mask = np.zeros_like(rgb_image_array[:, :, 0], dtype=np.uint8)
            polygon_mask = cv2.fillPoly(
                polygon_mask, [polygon_xy.astype(np.int32)], color=255
            )

            segmented_image = cv2.bitwise_and(
                rgb_image_array, rgb_image_array, mask=polygon_mask
            )
            segmented_depth = cv2.bitwise_and(depth_data, depth_data, mask=polygon_mask)

            if debug:

                fig, ax = plt.subplots()

                ax.imshow(rgb_image_array)

                ax.add_patch(polygon_patch)

                ax.set_xlim(0, rgb_image_array.shape[1])
                ax.set_ylim(rgb_image_array.shape[0], 0)

                plt.show()

                plt.imshow(segmented_image)
                plt.axis("off")
                plt.show()
                plt.imshow(segmented_depth)
                plt.axis("off")
                plt.show()
        return segmented_image, segmented_depth, prediction[0].boxes[0].xywh

    def segment(self, rgb_image, depth_data, debug=False):
        "Function to segment the plate in the image and return the segmented image and depth data."
        segmented_image = None
        segmented_depth = None
        rgb_image_array = np.asarray(rgb_image)
        h, w = rgb_image_array.shape[:2]
        prediction = self.model(rgb_image_array)

        if prediction[0].masks != None:
            polygon_xy = prediction[0].masks[0].xy[0]

            polygon_patch = Polygon(
                polygon_xy, closed=True, edgecolor="r", linewidth=2, fill=False
            )

            polygon_mask = np.zeros_like(rgb_image_array[:, :, 0], dtype=np.uint8)
            polygon_mask = cv2.fillPoly(
                polygon_mask, [polygon_xy.astype(np.int32)], color=255
            )

            segmented_image = cv2.bitwise_and(
                rgb_image_array, rgb_image_array, mask=polygon_mask
            )
            segmented_depth = cv2.bitwise_and(depth_data, depth_data, mask=polygon_mask)

            if debug:

                fig, ax = plt.subplots()

                ax.imshow(rgb_image_array)

                ax.add_patch(polygon_patch)

                ax.set_xlim(0, rgb_image_array.shape[1])
                ax.set_ylim(rgb_image_array.shape[0], 0)

                plt.show()

                plt.imshow(segmented_image)
                plt.axis("off")
                plt.show()
                plt.imshow(segmented_depth)
                plt.axis("off")
                plt.show()
        return segmented_image, segmented_depth

    def segment_and_conf(self, rgb_image, depth_data, debug=False):
        "Function to segment the plate in the image and return the segmented image, segmented depth data and the confidence score"
        segmented_image = None
        segmented_depth = None
        rgb_image_array = np.asarray(rgb_image)
        h, w = rgb_image_array.shape[:2]
        prediction = self.model(rgb_image_array)

        if prediction[0].masks != None:
            polygon_xy = prediction[0].masks[0].xy[0]

            polygon_patch = Polygon(
                polygon_xy, closed=True, edgecolor="r", linewidth=2, fill=False
            )

            polygon_mask = np.zeros_like(rgb_image_array[:, :, 0], dtype=np.uint8)
            polygon_mask = cv2.fillPoly(
                polygon_mask, [polygon_xy.astype(np.int32)], color=255
            )

            segmented_image = cv2.bitwise_and(
                rgb_image_array, rgb_image_array, mask=polygon_mask
            )
            segmented_depth = cv2.bitwise_and(depth_data, depth_data, mask=polygon_mask)

            if debug:

                fig, ax = plt.subplots()

                ax.imshow(rgb_image_array)

                ax.add_patch(polygon_patch)

                ax.set_xlim(0, rgb_image_array.shape[1])
                ax.set_ylim(rgb_image_array.shape[0], 0)

                plt.show()

                plt.imshow(segmented_image)
                plt.axis("off")
                plt.show()
                plt.imshow(segmented_depth)
                plt.axis("off")
                plt.show()
        return segmented_image, segmented_depth, prediction[0].boxes.conf
