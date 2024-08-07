#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


import copy
import os
import tf2_ros

from matplotlib import pyplot as plt
from position_controller import PositionControllerUR
import utils

import csv

from plate_detector import PlateDetector

from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from ur_master_thesis.msg import PlateROI

from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

"""
Node for gradient descent 
Initially records values of the objective function by visiting camera pses around the plate 
Then the initial points are used to fit the model 
With that model gradient ascent is performed by updating the model after every optimization step
"""

class OFRecorder:
    def __init__(self):
        self.experiment = "run1"  # for result collection
        rgb_topic = "/camera/color/image_raw"
        img_type = Image

        self.rgb_image = None

        self.bridge = CvBridge()
        rospy.Subscriber(rgb_topic, img_type, self.rgb_image_callback)

        self.frame1 = None
        self.frame2 = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.plate_detector = PlateDetector("object-detection/model/best.pt")

        self.tx = 0
        self.ty = 0
        self.tz = 0

        self.qx = 0
        self.qy = 0
        self.qz = 0
        self.qw = 0

        self.position_controller = PositionControllerUR()
        self.start_recording = True

        # Polynomial Model
        self.polynomial_model = Pipeline(
            [
                (
                    "polynomialfeatures",
                    PolynomialFeatures(degree=3),
                ),  # Change degree as needed
                ("linearregression", LinearRegression()),
            ]
        )

        # Gaussian Process Model
        kernel = WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)) + RBF(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e5)
        )
        # kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e7)) + WhiteKernel()
        # kernel = 1.0 *  WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)) # RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e5)) #

        self.gaussian_model = GaussianProcessRegressor(kernel=kernel, random_state=0)
        self.X = np.array([])
        self.Y = np.array([])
        self.X_pca = np.array([])
        self.PCA = PCA(n_components=4)
        self.scaler = MinMaxScaler()

        self.plate_center_available = True
        rospy.Subscriber("/plate/center", PlateROI, self.point_callback)
        rospy.Subscriber("/record/status", Bool, self.record_callback)

        self.iteration = 0
        self.obj_fun_iteration = 0
        self.histogram_it = 0

    def rgb_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb_image = cv_image
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def calculate_average_brightness(self, gray_image):
        average_brightness = np.mean(gray_image)
        return average_brightness

    def segment_img(self, rgb_img):
        height, width = rgb_img.shape[:2]
        depth_data_copy = np.ones((height, width), dtype=np.uint8)
        segmented_image, segmented_depth, bbox, bbox_conf = (
            self.plate_detector.segment_and_detect(
                rgb_img, depth_data_copy, debug=False
            )
        )
        return segmented_image, bbox

    def calculate_highlights(self, img_masked, img):

        histogram, bin_edges = np.histogram(img_masked, bins=256, range=(0, 256))

        smoothed_histogram = gaussian_filter1d(histogram, sigma=2)
        peaks, _ = find_peaks(smoothed_histogram, distance=10)

        if len(peaks) > 0:
            threshold = peaks[-1]
        else:
            rospy.logerr("Error in Histogram calculation")
            threshold = 0

        max_peak = np.argmax(smoothed_histogram)
        threshold_max = bin_edges[max_peak]

        num_peaks = 50
        peaks = np.argpartition(histogram, -num_peaks)[-num_peaks:]
        peaks = np.sort(peaks)

        above_threshold_mask = img > threshold

        highlighted_image = np.ones_like(img) * 255
        highlighted_image[~above_threshold_mask] = 0  # Set all other pixels to 0
        colors = {
            0: (231.999 / 255, 63.010 / 255, 72.012 / 255),
            1: (0.000, 135.992 / 255, 52.989 / 255),
            2: (251.991 / 255, 117.989 / 255, 51.995 / 255),
            3: (120.997 / 255, 35.011 / 255, 142.009 / 255),
        }

        #################### DEBUGGING ##############
        plt.figure(1)
        plt.clf()
        plt.plot(smoothed_histogram, color=colors[0])
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.title("Histogram")

        plt.savefig(f"/histogram_{self.histogram_it}.png")
        plt.figure(2)
        plt.clf()  #
        plt.plot(
            bin_edges[peaks],
            smoothed_histogram[peaks],
            "x",
            color=colors[0],
            label="Peaks",
        )
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.title("Histogram Smooth")

        plt.savefig(f"/histogramSmooth_{self.histogram_it}.png")
        ####################################
        self.histogram_it += 1
        return highlighted_image

    def get_roi_ocr(self, rgb_img):
        if self.plate_center_available:
            self.IS_EMPTY = True

            segmented_image, segmented_depth = self.plate_detector.segment(
                rgb_img, np.zeros_like(rgb_img[:, :, 0]), debug=False
            )
            if segmented_image is not None:
                gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

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
                self.plate_center_available = False
                return
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
            return has_chars

    def is_in_sector(self, top_left, bottom_right, sector):
        return top_left[1] >= sector[0] and bottom_right[1] <= sector[1]

    def perform_tempplate_matching(self, image):
        template_dir = "/char_templates_old/"

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

    def perform_image_preprocessing(self, image):

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        sharpened_image = cv2.filter2D(image, -1, kernel)
        img_brightness = self.calculate_brightness(sharpened_image)

        sharpened_image = 255 - (sharpened_image)

        a, b = sharpened_image.shape

        sharpened_image_resized = cv2.resize(sharpened_image, (b * 4, a * 4))

        sharpened_image_resized = self.bilateral_filter(sharpened_image_resized)
        return sharpened_image_resized

    def write_to_csv(self, data):
        filename = f"/{self.experiment}/record.csv"
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)

            writer.writerow(data)

    def create_roi_mask(self, image, roi_corners):
        image_gray = image
        mask = np.zeros(image_gray.shape, dtype=np.uint8)

        roi_corners = np.array(roi_corners, dtype=np.int32)

        mask = cv2.fillPoly(mask, [roi_corners], 255)

        image_masked = cv2.bitwise_and(image_gray, image_gray, mask=mask)
        return image_masked

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def calculate_distance_from_border(self, bbox, image_width, image_height):
        x, y, width, height = bbox[0].cpu().numpy()
        distance_left = x
        distance_right = image_width - (x + width)
        distance_top = y
        distance_bottom = image_height - (y + height)
        margin = 10
        x_margin_condition = x >= margin and x + width <= image_width - margin

        y_margin_condition = y >= margin and y + height <= image_height - margin

        if x_margin_condition:
            return 1, 1, 1, 1, True
        else:
            return distance_left, distance_right, distance_top, distance_bottom, False

    def calculate_percentage_from_border(self, bbox, image_width, image_height):
        distance_left, distance_right, distance_top, distance_bottom, indicator = (
            self.calculate_distance_from_border(bbox, image_width, image_height)
        )
        if indicator == False:
            total_distance = image_width + image_height
            percentage_left = distance_left / total_distance
            percentage_right = distance_right / total_distance
            percentage_top = distance_top / total_distance
            percentage_bottom = distance_bottom / total_distance
            percentages = np.array(
                [percentage_left, percentage_right, percentage_top, percentage_bottom]
            )
            min_perc = self.tanh(np.min(percentages))

            return min_perc
        else:
            return 1

    def calc_objective_function(self, segmented_img, roi, bbox):
        # we want a nice smooth differentiable function here
        # we need some kind of measur000ement how visible the object is
        image_height, image_width, _ = segmented_img.shape
        min_distance = self.calculate_percentage_from_border(
            bbox, image_width, image_height
        )

        rospy.logerr(min_distance)

        alpha = 0.5
        beta = 0.0
        gamma = 0.7
        feta = 0

        gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        mask = gray > 0

        image_cut = gray[mask]

        sum_non_zero_pixels = np.sum(gray[mask])
        num_non_zero_pixels = np.sum(mask)

        if num_non_zero_pixels > 0:
            gray_normalized = sum_non_zero_pixels / num_non_zero_pixels
        else:
            gray_normalized = 0

        highlighted_image = self.calculate_highlights(image_cut, gray)

        num_non_zero_pixels = np.sum(highlighted_image > 0)

        sum_highlighted_pixels = np.sum(highlighted_image)
        if num_non_zero_pixels > 0:
            highlighted_image_normalized = sum_highlighted_pixels / num_non_zero_pixels
        else:
            highlighted_image_normalized = 0

        obj_fun_sum = alpha * gray_normalized + beta * highlighted_image_normalized

        segmented_img_gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

        roi_mask = self.create_roi_mask(segmented_img_gray, roi)
        roi_mask_highlighted = self.create_roi_mask(highlighted_image, roi)
        num_non_zero_pixels = np.sum(roi_mask > 0)

        sum_roi_pixels = np.sum(roi_mask)
        if num_non_zero_pixels > 0:
            roi_mask_normalized = sum_roi_pixels / num_non_zero_pixels
        else:
            roi_mask_normalized = 0
        num_non_zero_pixels = np.sum(roi_mask_highlighted > 0)

        sum_roi_pixels = np.sum(roi_mask_highlighted)
        if num_non_zero_pixels > 0:
            roi_mask_normalized_highlighted = sum_roi_pixels / num_non_zero_pixels
        else:
            roi_mask_normalized_highlighted = 0

        obj_fun_img = (
            obj_fun_sum
            + gamma * roi_mask_normalized
            + 0 * roi_mask_normalized_highlighted
        )
        obj_fun_img = obj_fun_img + feta * min_distance
        combined_img = (
            alpha * gray
            + alpha * highlighted_image
            + beta * roi_mask
            + gamma * roi_mask_highlighted
        )
        # Result Collection
        plt.figure(1)
        plt.imshow(gray, cmap="gray")
        plt.imsave(
            f"/{self.experiment}/gray_img__record{self.obj_fun_iteration}.jpg", gray
        )
        plt.figure(2)
        plt.imshow(highlighted_image, cmap="gray")
        plt.imsave(
            f"/{self.experiment}/highlited_img_{self.obj_fun_iteration}.jpg",
            highlighted_image,
        )
        plt.figure(3)
        plt.imshow(roi_mask_highlighted, cmap="gray")
        plt.imsave(
            f"/{self.experiment}/roi_mask_highlighted_{self.obj_fun_iteration}.jpg",
            roi_mask_highlighted,
        )
        plt.figure(4)
        plt.imshow(roi_mask, cmap="gray")
        plt.imsave(
            f"/{self.experiment}/roi_mask_{self.obj_fun_iteration}.jpg", roi_mask
        )
        plt.figure(5)
        plt.imshow(combined_img, cmap="gray")
        plt.imsave(
            f"/{self.experiment}/combined_img_{self.obj_fun_iteration}.jpg",
            combined_img,
        )
        self.obj_fun_iteration += 1
        return obj_fun_img, combined_img, gray

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

    def get_roi(self, current_ee_pose):
        try:
            (
                point_center_edge_1,
                point_center_edge_2,
                point_center_edge_1_inner,
                point_center_edge_2_inner,
            ) = self.unpack_vectors()
            image_height, image_width, _ = self.rgb_image.shape
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
            roi = np.array(
                [
                    p_img_edge_1[0:2],
                    p_img_edge_2[0:2],
                    p_img_inner_2[0:2],
                    p_img_inner_1[0:2],
                ]
            )
            return roi
        except:
            return None

    def fit_polynomial_model(self):

        if self.start_recording == True and self.plate_center_available == True:

            ee_pose = self.get_ee_pose()

            rgb_img = copy.deepcopy(self.rgb_image)
            roi = self.get_roi(ee_pose)

            if roi is not None:
                segmented_image, bbox = self.segment_img(rgb_img)
                output_value, combined_img, gray = self.calc_objective_function(
                    segmented_image, roi, bbox
                )
                visisbility = self.get_roi_ocr(rgb_img)

                r = Rotation.from_quat(
                    [
                        ee_pose.rotation.x,
                        ee_pose.rotation.y,
                        ee_pose.rotation.z,
                        ee_pose.rotation.w,
                    ]
                )

                # Needed for additional experiments
                roll, pitch, yaw = r.as_euler("xyz", degrees=False)
                # New data point

                new_X = np.array(
                    [
                        ee_pose.translation.x,
                        ee_pose.translation.y,
                        ee_pose.translation.z,
                        ee_pose.rotation.x,
                        ee_pose.rotation.y,
                        ee_pose.rotation.z,
                        ee_pose.rotation.w,
                    ]
                )
                new_y = output_value

                self.X = (
                    np.vstack([self.X, new_X]) if self.X.size else np.array([new_X])
                )
                self.Y = np.append(self.Y, new_y)

                try:
                    rospy.loginfo("Start fit transform")
                    pose_array_scaled = self.scaler.fit_transform(self.X)

                    X_pca = self.PCA.fit_transform(pose_array_scaled)
                    f_point_prev = np.array([0])
                    try:
                        f_point_prev = self.polynomial_model.predict([X_pca[-1, :]])
                    except Exception as e:
                        rospy.logdebug("Waiting for more datapoints to fit the model")

                    self.polynomial_model = self.polynomial_model.fit(X_pca, self.Y)

                    f_point_post = self.polynomial_model.predict([X_pca[-1, :]])

                    csv_array = np.array(
                        [
                            self.iteration,
                            ee_pose.translation.x,
                            ee_pose.translation.y,
                            ee_pose.translation.z,
                            ee_pose.rotation.x,
                            ee_pose.rotation.y,
                            ee_pose.rotation.z,
                            ee_pose.rotation.w,
                            output_value,
                            f_point_prev[0],
                            f_point_post[0],
                            visisbility,
                            X_pca[-1, 0],
                            X_pca[-1, 1],
                            X_pca[-1, 2],
                            X_pca[-1, 3],
                        ]
                    )

                    self.write_to_csv(csv_array)
                except Exception as e:
                    rospy.logerr("We are not ready yet...")
                    print(e)
                cv2.imwrite(
                    f"/{self.experiment}/obj_img_{self.iteration}.jpg", combined_img
                )
                cv2.imwrite(f"/{self.experiment}/gray_img_{self.iteration}.jpg", gray)

                self.iteration = self.iteration + 1

            else:
                rospy.logerr("No ROI available for gradient descent")

    def plot_model_poly(self):

        x_range = np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), 50)
        y_range = np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)

        z_mean = np.mean(self.X[:, 2])
        roll_mean = np.mean(self.X[:, 3])
        pitch_mean = np.mean(self.X[:, 4])
        yaw_mean = np.mean(self.X[:, 5])

        z_grid = np.full_like(X_grid, z_mean)
        roll_grid = np.full_like(X_grid, roll_mean)
        pitch_grid = np.full_like(X_grid, pitch_mean)
        yaw_grid = np.full_like(X_grid, yaw_mean)

        XYZRPY_grid_transformed = np.stack(
            [X_grid, Y_grid, z_grid, roll_grid, pitch_grid, yaw_grid], axis=-1
        )

        XYZRPY_grid_transformed_flat = XYZRPY_grid_transformed.reshape(-1, 6)

        objective_function_values = self.polynomial_model.predict(
            XYZRPY_grid_transformed_flat
        ).reshape(X_grid.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            X_grid,
            Y_grid,
            objective_function_values,
            color="b",
            alpha=0.5,
            rstride=100,
            cstride=100,
        )

        ax.scatter(self.X[:, 0], self.X[:, 1], self.Y, color="r")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Objective Function")

        plt.savefig(f"/model_{self.iteration}.png")
        self.iteration = self.iteration + 1

    def fit_gaussian_model(self):
        if self.plate_center_available == True:

            ee_pose = self.get_ee_pose()

            rgb_img = copy.deepcopy(self.rgb_image)
            roi = self.get_roi(ee_pose)

            segmented_image, bbox = self.segment_img(rgb_img)
            output_value = self.calc_objective_function(segmented_image, roi)

            r = Rotation.from_quat(
                [
                    ee_pose.rotation.x,
                    ee_pose.rotation.y,
                    ee_pose.rotation.z,
                    ee_pose.rotation.w,
                ]
            )

            roll, pitch, yaw = r.as_euler("xyz", degrees=False)

            new_X = np.array(
                [
                    ee_pose.translation.x,
                    ee_pose.translation.y,
                    ee_pose.translation.z,
                    roll,
                    pitch,
                    yaw,
                ]
            )
            new_y = output_value

            self.X = np.vstack([self.X, new_X]) if self.X.size else np.array([new_X])
            self.Y = np.append(self.Y, new_y)

            X_transformed = self.scaler.fit_transform(self.X)

            self.gaussian_model = self.gaussian_model.fit(X_transformed, self.Y)
            rospy.loginfo("------------ Fit Gaussian Model ------------")

            self.plot_model()

    def plot_model(self):

        fixed_dims = np.mean(self.X[:, 2:], axis=0)

        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        x_values, y_values = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )

        X_meshgrid = np.c_[x_values.ravel(), y_values.ravel()]
        fixed_dims_repeated = np.tile(fixed_dims, (X_meshgrid.shape[0], 1))
        X_full = np.hstack((X_meshgrid, fixed_dims_repeated))

        y_pred, sigma = self.gaussian_model.predict(X_full, return_std=True)

        plt.figure(figsize=(10, 8))

        contour = plt.contourf(
            x_values,
            y_values,
            y_pred.reshape(x_values.shape),
            alpha=0.8,
            cmap="viridis",
        )
        plt.colorbar(contour, label="Predicted Data")

        scatter = plt.scatter(
            self.X[:, 0],
            self.X[:, 1],
            c=self.Y,
            edgecolor="k",
            cmap="coolwarm",
            label="Sampled Data",
        )
        plt.colorbar(scatter, label="Sampled Data")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Gaussian Process Model in X-Y Plane")
        plt.legend()

        plt.savefig(f"/gaussian_model/model_{self.iteration}.png")

        self.iteration = self.iteration + 1

    def get_camera_pose(self):
        try:

            transform = self.tf_buffer.lookup_transform(
                "base", "camera_color_optical_frame", rospy.Time(0), rospy.Duration(1.0)
            )
            return transform.transform
        except:
            return None

    def record_callback(self, msg):
        try:

            self.start_recording = msg.data
            if self.start_recording == False:
                self.plate_center_available = False
        except Exception as e:
            rospy.logerr("Error record callback: %s", str(e))

    def point_callback(self, msg):

        try:

            self.roi = msg
            self.plate_center_available = True
        except Exception as e:
            rospy.logerr("Error point callback: %s", str(e))

    def create_vec(self, point):
        vec = Point()
        vec.x = point[0]
        vec.y = point[1]

        vec.z = point[2]
        return vec


if __name__ == "__main__":
    rospy.init_node("obj_fun_recorder", anonymous=True)
    obj_fun_recorder = OFRecorder()

    rospy.sleep(0.2)

    try:
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            obj_fun_recorder.fit_polynomial_model()
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")
