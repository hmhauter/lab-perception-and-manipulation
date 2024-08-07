#!/usr/bin/env python3

import numpy as np
import cv2

from matplotlib import pyplot as plt


class OCR_ROI:
    def __init__(self) -> None:
        pass

    def getAngle(self, segmentation_mask):
        segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)

        _, binary_mask = cv2.threshold(segmentation_mask, 2, 255, cv2.THRESH_BINARY)

        kernel = np.ones((2, 2), np.uint8)

        eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        approx_contours = [
            cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
            for cnt in contours
        ]

        max_contour = max(approx_contours, key=cv2.contourArea)

        rotated_rect = cv2.minAreaRect(max_contour)

        angle = rotated_rect[-1]

        rect_points = cv2.boxPoints(rotated_rect)
        rect_points = np.int0(rect_points)

        angle_rad = np.radians(angle)
        vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        magnitude = np.linalg.norm(vector)

        unit_vector = vector / magnitude

        return angle, unit_vector

    def getLines(self, rgb_img, segmentation_mask):

        _, binary_mask = cv2.threshold(segmentation_mask, 2, 255, cv2.THRESH_BINARY)
        plt.figure(0)
        plt.imshow(binary_mask)

        kernel = np.ones((2, 2), np.uint8)

        eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
        plt.figure(1)
        plt.imshow(dilated_mask)
        plt.show()

        contours, _ = cv2.findContours(
            dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        approx_contours = [
            cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
            for cnt in contours
        ]

        max_contour = max(approx_contours, key=cv2.contourArea)

        rotated_rect = cv2.minAreaRect(max_contour)

        angle = rotated_rect[-1]

        rect_points = cv2.boxPoints(rotated_rect)
        rect_points = np.int0(rect_points)

        original_image = rgb_img
        cv2.drawContours(rgb_img, [rect_points], 0, (0, 255, 0), 2)

        (h, w) = original_image.shape[:2]
        center = (w // 2, h // 2)
        length = 100
        end_point = (
            int(center[0] + length * np.cos(np.radians(angle))),
            int(center[1] + length * np.sin(np.radians(angle))),
        )
        cv2.line(original_image, center, end_point, (0, 255, 0), 2)

        cv2.imshow("Image with Angle Line", rgb_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def getROI(self, rgb_img, segmentation_mask):

        gray_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)

        _, binary_mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)

        plt.figure(1)
        plt.imshow(rgb_img)
        plt.figure(2)
        plt.imshow(segmentation_mask)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        plt.figure(3)
        plt.imshow(contour_image)

        oriented_bounding_boxes = [cv2.minAreaRect(cnt) for cnt in contours]

        largest_box = max(
            oriented_bounding_boxes, key=lambda box: box[1][0] * box[1][1]
        )

        center, size, angle = largest_box

        a, b = size
        width = max(a, b)
        height = min(a, b)

        box_points = cv2.boxPoints(largest_box).astype(int)
        cv2.drawContours(rgb_img, [box_points], 0, (0, 255, 0), 1)

        angle_rad = np.radians(angle)

        roi_factor = 0.17 * width

        displacement_x = int(0.5 * width * np.cos(angle_rad))
        displacement_y = int(0.5 * width * np.sin(angle_rad))
        displacement_x_n = int((0.5 * width - roi_factor) * np.cos(angle_rad))
        displacement_y_n = int((0.5 * width - roi_factor) * np.sin(angle_rad))

        part_x = np.cos(angle_rad) * 0.5 * height
        part_y = np.sin(angle_rad) * 0.5 * height

        end_point_1 = (
            int(center[0] + displacement_x - part_x),
            int(center[1] - displacement_y - part_y),
        )
        end_point_2 = (
            int(center[0] + displacement_x + part_x),
            int(center[1] - displacement_y + part_y),
        )

        end_point_1_n = (
            int(center[0] + displacement_x_n - part_x),
            int(center[1] - displacement_y_n - part_y),
        )
        end_point_2_n = (
            int(center[0] + displacement_x_n + part_x),
            int(center[1] - displacement_y_n + part_y),
        )

        cv2.circle(rgb_img, end_point_1, 5, (0, 255, 0), -1)
        cv2.circle(rgb_img, end_point_2, 5, (0, 255, 0), -1)
        cv2.circle(rgb_img, end_point_1_n, 5, (0, 255, 0), -1)
        cv2.circle(rgb_img, end_point_2_n, 5, (0, 255, 0), -1)

        plt.figure(4)
        plt.imshow(rgb_img)
        plt.show()


if __name__ == "__main__":

    ocr_roi = OCR_ROI()
    rgb_img = cv2.imread("test.jpg")
    segmentation_mask = cv2.imread("test_segm.jpg", cv2.IMREAD_GRAYSCALE)
    ocr_roi.getLines(rgb_img, segmentation_mask)
