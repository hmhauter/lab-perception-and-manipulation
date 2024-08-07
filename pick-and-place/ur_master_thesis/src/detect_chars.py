#!/usr/bin/env python3

import rospy
import cv2
import csv
import numpy as np
from sensor_msgs.msg import Image
import std_msgs.msg
from ur_master_thesis.msg import PlateROI
from cv_bridge import CvBridge
import std_msgs
from plate_detector import PlateDetector
import utils
import copy
import os
import tf2_ros
from matplotlib import pyplot as plt

"""
Node to detect Characters with grid search. 
Gets signal to start character recognition through publishing of PlateROI
Publishes True or False if characters were not visible (True) or visible (False)
"""


class CharDetector:
    def __init__(self) -> None:
        self.text_pub = rospy.Publisher(
            "detected_text", std_msgs.msg.String, queue_size=10
        )
        self.angle_pub = rospy.Publisher(
            "correction_angle", std_msgs.msg.Float32, queue_size=10
        )
        self.bridge = CvBridge()
        self.plate_detector = PlateDetector("object-detection/model/best.pt")
        rospy.Subscriber("camera/color/image_raw", Image, self.image_callback)
        rospy.Subscriber("/plate/center", PlateROI, self.point_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.roi = None
        self.rgb_img = None
        self.counter = 0

        self.plate_center_available = False

        self.IS_EMPTY = False

    def get_ee_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "base", "tool0_controller", rospy.Time(0), rospy.Duration(1.0)
            )
            return transform.transform
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr("Failed to get transform: %s", e)

    def vector2array(self, vec):
        return np.array([vec.x, vec.y, vec.z])

    def unpack_vectors(self):
        msg = self.roi
        pt1 = self.vector2array(msg.point1)
        pt2 = self.vector2array(msg.point2)
        pt3 = self.vector2array(msg.point3)
        pt4 = self.vector2array(msg.point4)

        return pt1, pt2, pt3, pt4

    def chop_image(sellf, image):
        # Ign½ore the first 20 pixels from the top and bottom
        height, width = image.shape[:2]
        margin = 0
        cropped_image = image[margin : height - margin, :]

        strip_height = cropped_image.shape[0]

        subimage_height = strip_height // 8

        subimages = [
            cropped_image[i * subimage_height : (i + 1) * subimage_height, :]
            for i in range(8)
        ]
        return subimages

    def write_to_csv(self, row):

        filename = "/home/apo/catkin_ws/src/ur_master_thesis/src/full_plate.csv"
        with open(filename, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([row])

    def get_roi(self):
        if self.plate_center_available:
            self.IS_EMPTY = True

            rgb_img = copy.deepcopy(self.rgb_img)
            segmented_image, segmented_depth = self.plate_detector.segment(
                rgb_img, np.zeros_like(rgb_img[:, :, 0]), debug=False
            )

            if segmented_image is not None:
                gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Gray Image fom OCR", gray_image)
            cv2.waitKey(1)

            (
                point_center_edge_1,
                point_center_edge_2,
                point_center_edge_1_inner,
                point_center_edge_2_inner,
            ) = self.unpack_vectors()
            image_height, image_width = gray_image.shape
            current_ee_pose = self.get_ee_pose()
            p_img_edge_1 = utils.transform_base_img(
                point_center_edge_1, current_ee_pose, image_width, image_height
            )
            p_img_edge_2 = utils.transform_base_img(
                point_center_edge_2, current_ee_pose, image_width, image_height
            )
            p_img_inner_1 = utils.transform_base_img(
                point_center_edge_1_inner, current_ee_pose, image_width, image_height
            )
            p_img_inner_2 = utils.transform_base_img(
                point_center_edge_2_inner, current_ee_pose, image_width, image_height
            )
            if (
                p_img_edge_1 is None
                or p_img_edge_2 is None
                or p_img_inner_1 is None
                or p_img_inner_2 is None
            ):

                return
            try:
                cropped_img = utils.cut_out_img(
                    gray_image,
                    p_img_edge_1[0:2],
                    p_img_edge_2[0:2],
                    p_img_inner_1[0:2],
                    p_img_inner_2[0:2],
                )
                cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                img_preprocessed = self.perform_image_preprocessing(cropped_img)
                img_for_showing = copy.deepcopy(img_preprocessed)
                img_for_showing = cv2.resize(img_for_showing, (90, 600))

                cv2.imshow("Image preprocessed for OCR", img_for_showing)
                cv2.waitKey(1)

                has_chars = self.perform_tempplate_matching(img_preprocessed)
                if has_chars == 1:
                    self.IS_EMPTY = False
                    rospy.loginfo("Chars detected on Plate")

                send_empty = std_msgs.msg.String()
                send_empty.data = str(self.IS_EMPTY)
                self.text_pub.publish(send_empty)
            except Exception as e:
                rospy.logerr("Error processing ROI: %s", str(e))

    def calculate_brightness(self, gray):
        mask = gray > 0
        unmasked_brightness = gray[mask]

        if unmasked_brightness.size == 0:
            rospy.logwarn("Warning: unmasked_brightness array is empty.")
            brightness = 255
        else:
            brightness = unmasked_brightness.mean()

            if np.isnan(brightness):
                rospy.logwarn("Warning: Calculated mean is NaN.")
                brightness = 255

        return int(brightness)

    def bilateral_filter(self, image):
        # d: Diameter of each pixel neighborhood
        # sigmaColor: Filter sigma in the color space
        # sigmaSpace: Filter sigma in the coordinate space
        smoothed_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        return smoothed_image

    def is_in_sector(self, top_left, bottom_right, sector):
        return top_left[1] >= sector[0] and bottom_right[1] <= sector[1]

    def check_results(self, results):

        empty_count = 0

        num_results = (
            len(results["sector1"]) + len(results["sector2"]) + len(results["sector3"])
        )

        # Check each sector for emptiness
        if not results["sector1"]:
            empty_count += 1
        if not results["sector2"]:
            empty_count += 1
        if not results["sector3"]:
            empty_count += 1

        if empty_count >= 2 or num_results < 3:
            return 0
        else:
            return 1

    def visualize_detections(self, image, detections):
        height, width = image.shape
        margin = 50
        template_size = 45
        canvas_height = height + 2 * margin
        canvas_width = width + (template_size + 4 * margin)

        canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255
        canvas[margin : margin + height, margin : margin + width] = image

        for i, (template, top_left, bottom_right) in enumerate(detections):
            template_x = width + 2 * margin
            template_y = margin + i * (template_size + margin)

            canvas[
                template_y : template_y + template_size,
                template_x : template_x + template_size,
            ] = template
            cv2.rectangle(
                canvas,
                (template_x, template_y),
                (template_x + template_size, template_y + template_size),
                0,
                2,
            )

            line_start = (template_x + template_size // 2, template_y + template_size)
            line_end = (top_left[0] + margin, top_left[1] + margin)
            cv2.line(canvas, line_start, line_end, 0, 2)
            cv2.rectangle(
                canvas,
                (top_left[0] + margin, top_left[1] + margin),
                (bottom_right[0] + margin, bottom_right[1] + margin),
                0,
                2,
            )

        plt.figure()
        plt.grid(False)
        plt.axis("off")
        plt.imshow(canvas, cmap="gray")
        plt.title("Template Matching - Failure Analysis")
        plt.show()

    def perform_tempplate_matching(self, image):
        template_dir = "/char_templates/"

        template_list = sorted(os.listdir(template_dir))

        height, width = image.shape

        sector1 = (0, height // 3)
        sector2 = (height // 3, 2 * height // 3)
        sector3 = (2 * height // 3, height)

        results = {"sector1": [], "sector2": [], "sector3": []}
        all_detections = []

        for template_name in template_list:

            template = cv2.imread(os.path.join(template_dir, template_name), 0)
            template = cv2.resize(template, (45, 45))

            w, h = template.shape[::-1]

            img = image.copy()
            method = cv2.TM_CCORR_NORMED

            res = cv2.matchTemplate(img, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.95:

                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)

                template_name_no_ext = template_name.replace(".png", "")

                # Check for Sector 1: A, B, C, D
                if template_name_no_ext in ["A", "B", "C", "D"]:
                    if self.is_in_sector(top_left, bottom_right, sector1):
                        results["sector1"].append(template_name_no_ext)

                # Check for Sector 2: C, D, E, F
                if template_name_no_ext in ["C", "D", "E", "F"]:
                    if self.is_in_sector(top_left, bottom_right, sector2):
                        results["sector2"].append(template_name_no_ext)

                # Check for Sector 3: E, F, G, H
                if template_name_no_ext in ["E", "F", "G", "H"]:
                    if self.is_in_sector(top_left, bottom_right, sector3):
                        results["sector3"].append(template_name_no_ext)

                all_detections.append((template, top_left, bottom_right))

        rospy.logdebug("Items detected in Sector 1:", results["sector1"])
        rospy.logdebug("Items detected in Sector 2:", results["sector2"])
        rospy.logdebug("Items detected in Sector 3:", results["sector3"])

        is_valid = self.check_results(results)

        if is_valid == 1:
            return True
        else:
            return False

    def perform_image_preprocessing(self, image):

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        sharpened_image = cv2.filter2D(image, -1, kernel)
        img_brightness = self.calculate_brightness(sharpened_image)

        if img_brightness < 100:
            sharpened_image = cv2.equalizeHist(sharpened_image)

        elif img_brightness <= 155 and img_brightness >= 100:
            sharpened_image = 255 - (sharpened_image)

        a, b = sharpened_image.shape

        sharpened_image_resized = cv2.resize(sharpened_image, (b * 4, a * 4))

        sharpened_image_resized = self.bilateral_filter(sharpened_image_resized)
        return sharpened_image_resized

    def unsharp_masking(self, image, sigma=1.2, strength=2.0):
        # Apply Gaussian blur to the image
        blurred = cv2.GaussianBlur(
            image, (0, 0), sigma
        )  # if 0 then computed from sigma
        # Subtract the blurred image from the original image
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

        return sharpened

    def point_callback(self, msg):

        try:

            self.roi = msg
            self.plate_center_available = True

        except Exception as e:
            rospy.logerr("Error processing chars: %s", str(e))

    def image_callback(self, msg):
        try:

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb_img = np.array(cv_image)

        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))


if __name__ == "__main__":
    rospy.loginfo("Start Detect Character Node")

    rospy.init_node("text_detection_node")
    detect_char = CharDetector()

    try:
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            detect_char.get_roi()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
