#!/usr/bin/env python3

import numpy as np 
import cv2

from matplotlib import pyplot as plt

class OCR_ROI:
    def __init__(self) -> None:
        pass

    def getAngle(self, segmentation_mask):
        segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to create a binary image
        _, binary_mask = cv2.threshold(segmentation_mask, 2, 255, cv2.THRESH_BINARY)

       # Perform morphological operations to enhance features (optional)
        kernel = np.ones((2, 2), np.uint8)
        # Apply erosion
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

        # Apply dilation
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate contours
        approx_contours = [cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True) for cnt in contours]

        # Find the contour with the maximum area (assuming it corresponds to the rectangular object)
        max_contour = max(approx_contours, key=cv2.contourArea)

        # Fit a rotated rectangle to the contour
        rotated_rect = cv2.minAreaRect(max_contour)

        # Extract angle of rotation from the rotated rectangle
        angle = rotated_rect[-1]

        # Get the vertices of the rotated rectangle
        rect_points = cv2.boxPoints(rotated_rect)
        rect_points = np.int0(rect_points)

        # Define the vector
        angle_rad = np.radians(angle)
        vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])  # Replace x and y with the components of your vector

        # Calculate the magnitude of the vector
        magnitude = np.linalg.norm(vector)

        # Normalize the vector to obtain the unit vector
        unit_vector = vector / magnitude

        return angle, unit_vector



    def getLines(self, rgb_img, segmentation_mask):

        # Apply thresholding to create a binary image
        _, binary_mask = cv2.threshold(segmentation_mask, 2, 255, cv2.THRESH_BINARY)
        plt.figure(0)
        plt.imshow(binary_mask)
       # Perform morphological operations to enhance features (optional)
        kernel = np.ones((2, 2), np.uint8)
        # Apply erosion
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

        # Apply dilation
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
        plt.figure(1)
        plt.imshow(dilated_mask)
        plt.show()
        # Find contours
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate contours
        approx_contours = [cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True) for cnt in contours]

        # Find the contour with the maximum area (assuming it corresponds to the rectangular object)
        max_contour = max(approx_contours, key=cv2.contourArea)

        # Fit a rotated rectangle to the contour
        rotated_rect = cv2.minAreaRect(max_contour)

        # Extract angle of rotation from the rotated rectangle
        angle = rotated_rect[-1]

        # Get the vertices of the rotated rectangle
        rect_points = cv2.boxPoints(rotated_rect)
        rect_points = np.int0(rect_points)

        # Draw the rotated rectangle on the original image
        original_image = rgb_img
        cv2.drawContours(rgb_img, [rect_points], 0, (0, 255, 0), 2)  # Draw green rectangle with thickness 2

        # Draw a line with the extracted angle on the original image
        (h, w) = original_image.shape[:2]
        center = (w // 2, h // 2)
        length = 100  # Length of the line
        end_point = (int(center[0] + length * np.cos(np.radians(angle))), int(center[1] + length * np.sin(np.radians(angle))))
        cv2.line(original_image, center, end_point, (0, 255, 0), 2)  # Draw green line with thickness 2

        # print("ANGLE")
        # print(angle)
        # Save or display the image with the drawn line
        # cv2.imwrite('image_with_angle_line.jpg', original_image)
        cv2.imshow('Image with Angle Line', rgb_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def getROI(self, rgb_img, segmentation_mask):
        # Convert the segmentation mask to grayscale
        gray_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to create a black-and-white segmentation mask
        _, binary_mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)

        plt.figure(1)
        plt.imshow(rgb_img)
        plt.figure(2)
        plt.imshow(segmentation_mask)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image to draw contours on
        contour_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        # Draw contours on the image
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        plt.figure(3)
        plt.imshow(contour_image)


        # Calculate the oriented bounding box for each contour
        oriented_bounding_boxes = [cv2.minAreaRect(cnt) for cnt in contours]

        # Select the rotated rectangle with the largest area
        largest_box = max(oriented_bounding_boxes, key=lambda box: box[1][0] * box[1][1])
        print("LARGEST BOX")
        print(largest_box)

        center, size, angle = largest_box
        print(center)
        print(size)
        print(angle)
        a, b = size
        width = max(a, b)
        height = min(a, b)

        # Draw the largest rotated rectangle on the original image
        box_points = cv2.boxPoints(largest_box).astype(int)
        cv2.drawContours(rgb_img, [box_points], 0, (0, 255, 0), 1)

        angle_rad = np.radians(angle)

        roi_factor = 0.17 * width

        # Calculate the displacement along the x-axis and y-axis
        displacement_x = int(0.5 * width * np.cos(angle_rad))
        displacement_y = int(0.5 * width * np.sin(angle_rad))
        displacement_x_n = int((0.5 * width-roi_factor) * np.cos(angle_rad))
        displacement_y_n = int((0.5 * width-roi_factor) * np.sin(angle_rad))

        part_x = np.cos(angle_rad) * 0.5 * height
        part_y = np.sin(angle_rad) * 0.5 * height

        # Calculate the coordinates of the end point of the line segment
        end_point_1 = (int(center[0] + displacement_x - part_x), int(center[1] - displacement_y - part_y))
        end_point_2 = (int(center[0] + displacement_x + part_x), int(center[1] - displacement_y + part_y))

        end_point_1_n = (int(center[0] + displacement_x_n - part_x), int(center[1] - displacement_y_n - part_y))
        end_point_2_n = (int(center[0] + displacement_x_n + part_x), int(center[1] - displacement_y_n + part_y))

        cv2.circle(rgb_img, end_point_1, 5, (0, 255, 0), -1) 
        cv2.circle(rgb_img, end_point_2, 5, (0, 255, 0), -1) 
        cv2.circle(rgb_img, end_point_1_n, 5, (0, 255, 0), -1) 
        cv2.circle(rgb_img, end_point_2_n, 5, (0, 255, 0), -1) 

        # # Crop the region of interest from the segmentation mask
        # roi = segmentation_mask[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

        plt.figure(4)
        plt.imshow(rgb_img)
        plt.show()


if __name__ == '__main__':
    print("START")
    ocr_roi = OCR_ROI()
    rgb_img = cv2.imread("/home/apo/catkin_ws/src/ur_master_thesis/src_new/ocr/5_img.jpg")
    segmentation_mask = cv2.imread("/home/apo/catkin_ws/src/ur_master_thesis/src_new/ocr/5_segmented_img.jpg", cv2.IMREAD_GRAYSCALE)
    ocr_roi.getLines(rgb_img, segmentation_mask)


