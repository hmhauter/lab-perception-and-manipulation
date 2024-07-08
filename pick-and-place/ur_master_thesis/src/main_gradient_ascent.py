#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import copy
import tf2_ros
import time
from matplotlib import pyplot as plt
from position_controller import PositionControllerUR
import utils
import transforms3d.quaternions as quat
import robotiq_gripper
import csv
from plate_detector import PlateDetector
from grasp_point_estimator_new import GraspPointEstimator
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from ur_master_thesis.msg import GraspPoints, PlateROI 
from scipy.spatial.transform import Rotation

from tf2_msgs.msg import TFMessage

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

class AutomationPipeline():
    def __init__(self):

        self.experiment = "run8"

        # DEBUG VALUES 
        self.DEBUG_RUN_NUM = 0
        self.DEBUG_FOLDER = "/home/apo/catkin_ws/src/ur_master_thesis/src/TESTING/test5/"
        self.DEBUG_WRITE = []

        UR_IP = "192.168.1.100"
        self.SEGMENT = True
        self.PLATE_ALERT = False

        self.MODE = 0

        self.LENGTH = 0.124
        self.WIDTH = 0.081

        self.depth_image = None
        self.rgb_image = None
        self.camera_info = None
        self.bridge = CvBridge()

        self.camera_intrinsics = {}

        self.SEARCH_MIN_BASE = -50
        self.SEARCH_MAX_BASE = 30

        self.SEARCH_MIN_WRIST = -20
        self.SEARCH_MAX_WRIST = 20

        self.ANGLE_INCREMENT = 2
        self.ANGLE_INCREMENT_WRIST = 3

        self.direction = -1
        self.direction_wrist = -1
        self.search_angle = 0

        self.SUCCESS_THRESHOLD = 3
        self.PICK_SUCCESS = 0

        self.gripper = robotiq_gripper.RobotiqGripper()
        rospy.loginfo("Connecting to gripper...")
        self.gripper.connect(UR_IP, 63352)
        rospy.loginfo("Activating gripper...")
        self.gripper.activate()

        self.aruco_position = None


        # Specify the topic name and message type
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        rgb_topic = "/camera/color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        img_type = Image
        info_type = CameraInfo

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber("/tf", TFMessage, self.tf_callback)
        self.center_point_pub = rospy.Publisher('/plate/center', PlateROI, queue_size=10)

        # Create a subscriber
        rospy.Subscriber(camera_info_topic, info_type, self.camera_info_callback)
        rospy.Subscriber(depth_topic, img_type, self.depth_image_callback)
        rospy.Subscriber(rgb_topic, img_type, self.rgb_image_callback)
        
        # fixed measurements
        self.OFFSET_Z_ABOVE = 0.17 + 0.025 # 0.1776 # m
        self.OFFSET_Y_ABOVE = -0.1 #0.01525#+0.002  # measured in CAD
        self.OFFSET_Y_SIDE = 0.05 # 0.047 # 0.05 
        self.OFFSET_Z_SIDE = 0.1413  # measured in CAD
        LENGTH = 0.124
        WIDTH = 0.081

        self.roi = None
        self.plate_detector = PlateDetector("/home/apo/Documents/MasterThesis/adaptive-lab-automation/datasets/runs/segment/train/weights/best.pt")
        self.grasp_point_estimator = GraspPointEstimator(WIDTH, LENGTH)
        self.position_controller = PositionControllerUR()

        self.polynomial_model = Pipeline([
            ('polynomialfeatures', PolynomialFeatures(degree=3)),  # Change degree as needed
            ('linearregression', LinearRegression())
        ])
        self.X = []
        self.Y = []
        self.X_pca = []
        self.PCA = PCA(n_components=4)
        self.scaler = MinMaxScaler()
        # self.temp_fit_initial()

        self.pose_logging = []

        self.iteration_gd = 0
        
        # self.grasp_points = GraspPoints()
        # self.grasp_point_pub = rospy.Publisher('/grasp/points',  GraspPoints, queue_size=1)
        self.record_pub = rospy.Publisher('/record/status', Bool, queue_size=1)

    # def temp_fit_initial(self):
    #     csv_file = "/home/apo/catkin_ws/src/ur_master_thesis/src/ObjectiveFunction/run3/fun3.csv"
    #     pose_array = []
    #     obj_array = []

    #     with open(csv_file, 'r') as file:
    #         csv_reader = csv.reader(file)
    #         for row in csv_reader:
    #             pose_array.append([float(row[1]), float(row[2]), float(row[3]), 
    #                             float(row[4]), float(row[5]), float(row[6]), float(row[7])])
    #             obj_array.append(float(row[8]))

    #     pose_array = np.array(pose_array)
    #     obj_array = np.array(obj_array)

    #     self.X = pose_array
    #     self.Y = obj_array

    #     pose_array_scaled = self.scaler.fit_transform(pose_array)       
    #     X_pca = self.PCA.fit_transform(pose_array_scaled)  
    #     self.polynomial_model = self.polynomial_model.fit(X_pca, obj_array)

    def write_to_csv(self, filename, data):
        # Open the CSV file in append mode
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the data to the CSV file
            writer.writerow(data)

    def get_ee_pose(self):
        try:
            # Wait for the transform to become available
            transform = self.tf_buffer.lookup_transform("base", "tool0_controller", rospy.Time(0), rospy.Duration(1.0))
            return transform.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to get transform: %s", e)

        
    def deliver(self, board_transformation):
            ee_pose = self.get_ee_pose()
            ee_quat = np.array([
                ee_pose.rotation.x,
                ee_pose.rotation.y,
                ee_pose.rotation.z,
                ee_pose.rotation.w
            ])

            original_quat = np.array([
                board_transformation.rotation.x,
                board_transformation.rotation.y,
                board_transformation.rotation.z,
                board_transformation.rotation.w
            ])

            # Convert quaternion to rotation matrix
            original_rot_matrix = Rotation.from_quat(original_quat).as_matrix()
            if self.MODE == 0:
                x_rotation_angle = 180
                final_translation = np.array([
                    board_transformation.translation.x,
                    board_transformation.translation.y,
                    board_transformation.translation.z + self.OFFSET_Z_ABOVE 
                ])
            else:
            # Define the rotation angle (+90 degrees around local X-axis)
                x_rotation_angle = 90  # Degrees
                final_translation = np.array([
                    board_transformation.translation.x,
                    board_transformation.translation.y + self.OFFSET_Z_SIDE,
                    board_transformation.translation.z + self.OFFSET_Y_SIDE + 0.02
                ])
            # Define the rotation matrix for rotation around the local X-axis
            x_rotation_matrix = Rotation.from_euler('x', x_rotation_angle, degrees=True).as_matrix()

            # Define the rotation angle (+90 degrees around local X-axis)
            y_rotation_angle = 180  # Degrees
            # Define the rotation matrix for rotation around the local X-axis
            y_rotation_matrix = Rotation.from_euler('z', y_rotation_angle, degrees=True).as_matrix()

            x_new_rot_matrix = np.dot(original_rot_matrix, x_rotation_matrix)
            xy_new_rot_matrix = np.dot(x_new_rot_matrix, y_rotation_matrix)

            x_new_quat = Rotation.from_matrix(x_new_rot_matrix).as_quat()
            xy_new_quat = Rotation.from_matrix(xy_new_rot_matrix).as_quat()

            # Calculate the angles between original and resulting quaternions
            q_delta1 = x_new_quat * ee_quat.conj()
            q_delta2 = xy_new_quat * ee_quat.conj()
            # Convert the quaternion to rotation angles
            angles1 = Rotation.from_quat(q_delta1).as_euler('xyz')
            angles2 = Rotation.from_quat(q_delta2).as_euler('xyz')
            angle1 = np.linalg.norm(angles1)
            angle2 = np.linalg.norm(angles2)
            print(angle1)
            print(angle2)
            # Determine which quaternion results in less rotation from the original position
            if angle1 < angle2:
                print("OPTION 1")
                final_quat = x_new_quat
            else:
                print("OPTION 2")
                final_quat = xy_new_quat

            return final_translation, final_quat

    
    def start(self):        
        rospy.loginfo("Start main script...")
        if self.PICK_SUCCESS == 0:
            rospy.loginfo("PICK MODE")
            self.DEBUG_RUN_NUM += 1
            self.start_time = time.time()
             
            self.DEBUG_WRITE.append(str(self.DEBUG_RUN_NUM))
            self.DEBUG_WRITE.append(self.MODE)
            
            self.position_controller.init_search()

            self.search_angle = 0
            self.search_angle_wrist = 0
            angle_counter = 0
            bbox = None
            is_inside = False
            detection_success = 0
            while bbox == None or detection_success <= self.SUCCESS_THRESHOLD or is_inside == False or self.aruco_position == None: 
                rospy.loginfo(f"what is the problem?")
                print(bbox)
                print(detection_success)
                print(is_inside)
                print(self.aruco_position)
                bbox = self.plate_detector.detect(self.rgb_image)
                if bbox != None:
                    detection_success += 1
                    rospy.loginfo(f"Bounding Box:  {bbox}")
                    bboxes = bbox.cpu().numpy()
                    is_inside, side, x_direction, y_direction, vertical_direction = utils.analyze_bounding_box(bboxes[0])
                    if is_inside == False and detection_success >= self.SUCCESS_THRESHOLD:
                        if x_direction == "too much" and side == "right":
                            self.direction = -1
                            self.position_controller.move_base_joint(self.ANGLE_INCREMENT * self.direction)
                        if x_direction == "too much" and side == "left":
                            self.direction = 1
                            self.position_controller.move_base_joint(self.ANGLE_INCREMENT * self.direction)
                        if y_direction == "too much" and vertical_direction == "up":
                            self.position_controller.move_wrist1_joint(1 * self.ANGLE_INCREMENT_WRIST)
                        if y_direction == "too much" and vertical_direction == "down":
                            self.position_controller.move_wrist1_joint(-1 * self.ANGLE_INCREMENT_WRIST)
                    else:
                        self.position_controller.move_base_joint(self.ANGLE_INCREMENT * self.direction)
                    
                else:
                    if detection_success > 0:
                        detection_success -= 1

                    if self.search_angle >= self.SEARCH_MAX_BASE or self.search_angle <= self.SEARCH_MIN_BASE:
                        # Change direction
                        self.direction *= -1
                        if self.direction == -1:
                            # MOVE WRIST 
                            if (self.search_angle_wrist >= self.SEARCH_MAX_WRIST or self.search_angle_wrist <= self.SEARCH_MIN_WRIST) and is_inside == False:
                            # Change direction
                                self.direction_wrist *= -1
                            self.position_controller.move_wrist1_joint(self.ANGLE_INCREMENT_WRIST * self.direction_wrist)
                            self.search_angle_wrist += self.ANGLE_INCREMENT_WRIST * self.direction_wrist
                    self.search_angle += self.ANGLE_INCREMENT * self.direction
                    self.position_controller.move_base_joint(self.ANGLE_INCREMENT * self.direction)
                    angle_counter += self.ANGLE_INCREMENT

            debug_array = self.run()
            self.DEBUG_WRITE.extend(debug_array)

        elif self.PICK_SUCCESS == 1:
            # DELIVER 
            rospy.loginfo("DELIVER MODE")
            # this should be an artifact that is not used any more
            while self.aruco_position == None: 
                print(self.search_angle)
                self.position_controller.move_base_joint(self.ANGLE_INCREMENT * self.direction)

                if self.search_angle >= self.SEARCH_MAX or self.search_angle <= self.SEARCH_MIN:
                    # Change direction
                    self.direction *= -1
                self.search_angle += self.ANGLE_INCREMENT * self.direction
            ######################
            deliver_timer = time.time()
            aruco_translation, aruco_quat = self.deliver(self.aruco_position)

            self.position_controller.go_to_pose(aruco_translation, aruco_quat)
            print("ForcE controL")
            rospy.sleep(1.0)
            deliver_flag = False
            deliver_flag = self.position_controller.deliver_plate()
            if deliver_flag:
                self.gripper.move_and_wait_for_pos(0, 10, 10)
                self.PICK_SUCCESS = 0
            
            # self.position_controller.remove_collision_object()
            stop_time = time.time()
            self.DEBUG_WRITE.append(str(stop_time-deliver_timer))
            self.DEBUG_WRITE.append(str(stop_time-self.start_time))
            print(self.DEBUG_WRITE)
            self.write_to_csv(self.DEBUG_FOLDER+"test.csv", self.DEBUG_WRITE)
            self.DEBUG_WRITE = []



    def grasp_above(self, normal_z, normal_y, ee_pose, point_base):
        R_tcp, z_tcp, y_tcp = utils.get_tcp_rot_above(normal_z, normal_y, ee_pose)
        quaternion = Rotation.from_matrix(R_tcp).as_quat()
        point_base[:3] -= self.OFFSET_Z_ABOVE * z_tcp
        point_base[:3] += self.OFFSET_Y_ABOVE * R_tcp[:, 1] 

        print("R TCP")
        print(R_tcp)

        print(point_base)

        return quaternion, point_base, R_tcp

    def grasp_side(self, normal_z, normal_y, ee_pose, point_base):
        R_tcp, z_tcp, y_tcp = utils.get_tcp_rot_side(normal_z, normal_y, ee_pose)
        quaternion = Rotation.from_matrix(R_tcp).as_quat()
        point_base[:3] += self.OFFSET_Z_SIDE * R_tcp[:, 2] 
        point_base[:3] -= self.OFFSET_Y_SIDE * y_tcp

        return quaternion, point_base, R_tcp


    def create_vec(self, point):
        vec = Point()
        vec.x = point[0]
        vec.y = point[1]
        vec.z = point[2]
        return vec
    
    def publish_plate_center(self, pt1, pt2, pt3, pt4):
        print("Publish plate center")
        plate_roi = PlateROI()
        vec1 = self.create_vec(pt1)
        vec2 = self.create_vec(pt2)
        vec3 = self.create_vec(pt3)
        vec4 = self.create_vec(pt4)

        plate_roi.point1 = vec1
        plate_roi.point2 = vec2
        plate_roi.point3 = vec3
        plate_roi.point4 = vec4

        rospy.loginfo("Publishing plate ROI points in robot base frame.")

        self.roi = plate_roi
        self.center_point_pub.publish(plate_roi)

############ GADIENT ASCENT FUNCTIONS ################

    def create_roi_mask(self, image, roi_corners):
        image_gray = image # cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(image_gray.shape, dtype=np.uint8)

        roi_corners = np.array(roi_corners, dtype=np.int32)


        mask = cv2.fillPoly(mask, [roi_corners], 255)
   
        image_masked = cv2.bitwise_and(image_gray, image_gray, mask=mask)
        return image_masked

    # def segment_img(self, rgb_img):
    #     print("Segment image...")
    #     print(rgb_img.shape)
    #     height, width = rgb_img.shape[:2]
    #     depth_data_copy = np.ones((height, width), dtype=np.uint8)
    #     segmented_image, _ = self.plate_detector.segment(rgb_img, depth_data_copy, debug=False)
    #     return segmented_image

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    def calculate_distance_from_border(self, bbox, image_width, image_height):
        x, y, width, height = bbox[0].cpu().numpy()
        distance_left = x
        distance_right = image_width - (x + width)
        distance_top = y
        distance_bottom = image_height - (y + height)
        margin = 20
        x_margin_condition = x >= margin and x + width <= image_width - margin
        y_margin_condition = y >= margin and y + height <= image_height - margin
        if x_margin_condition and y_margin_condition:
            return 1, 1, 1, 1, True
        else:
            return distance_left, distance_right, distance_top, distance_bottom, False

    def calculate_percentage_from_border(self, bbox, image_width, image_height):
        distance_left, distance_right, distance_top, distance_bottom, indicator = self.calculate_distance_from_border(bbox, image_width, image_height)
        if indicator == False:
            total_distance = image_width + image_height  # Total distance from left, right, top, and bottom combined
            percentage_left = (distance_left / total_distance) 
            percentage_right = (distance_right / total_distance) 
            percentage_top = (distance_top / total_distance) 
            percentage_bottom = (distance_bottom / total_distance) 
            percentages = np.array([percentage_left, percentage_right, percentage_top, percentage_bottom])
            min_perc = self.tanh(np.min(percentages))
            print("MIN AFTER TANH FUNC")
            print(min_perc)
            # result is between 0 and 1
            return min_perc
        else:
            return 1

    def calc_objective_function(self, segmented_img, roi, bbox):
        # we want a nice smooth differentiable function here
        # we need some kind of measur000ement how visible the object is 
        image_height, image_width, _ = segmented_img.shape
        min_distance = self.calculate_percentage_from_border(bbox, image_width, image_height)

        print("???????????????????????????????")
        rospy.logerr(min_distance)
        # and even more important how visible the ROI is 
        rospy.loginfo("Calculate objective function...")
        alpha = 0.5
        beta = 0.3
        gamma = 0.8
        feta = 60


        # Convert the image to grayscale
        print(segmented_img.shape)
        gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        # Apply the mask to extract the relevant parts of the grayscale image
        image_cut = gray[mask]

        # Get the number of not masked pixels
        # Calculate the sum of non-zero pixel values and the number of non-zero pixels
        sum_non_zero_pixels = np.sum(gray[mask])
        num_non_zero_pixels = np.sum(mask)

        # Normalize the sum of pixel values by the number of non-zero pixels
        if num_non_zero_pixels > 0:
            gray_normalized = sum_non_zero_pixels / num_non_zero_pixels
        else:
            gray_normalized = 0  # that means nothing was visible on the image
        print("##############")
        # Calculate highlights 
        highlighted_image = self.calculate_highlights(image_cut, gray)

        num_non_zero_pixels = np.sum(highlighted_image>0)
        # Normalize the highlighted image by the number of non-zero pixels
        sum_highlighted_pixels = np.sum(highlighted_image)
        if num_non_zero_pixels > 0:
            highlighted_image_normalized = sum_highlighted_pixels / num_non_zero_pixels
        else:
            highlighted_image_normalized = 0

        obj_fun_sum = alpha * gray_normalized + beta * highlighted_image_normalized 
        print("Segment gray")
        print(segmented_img.shape)
        segmented_img_gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        # now weight the pixels that are in the ROI more 
        roi_mask = self.create_roi_mask(segmented_img_gray, roi)
        roi_mask_highlighted = self.create_roi_mask(highlighted_image, roi)
        num_non_zero_pixels = np.sum(roi_mask>0)

        # Normalize the ROI mask by the number of non-zero pixels
        sum_roi_pixels = np.sum(roi_mask)
        if num_non_zero_pixels > 0:
            roi_mask_normalized = sum_roi_pixels / num_non_zero_pixels
        else:
            roi_mask_normalized = 0  
        num_non_zero_pixels = np.sum(roi_mask_highlighted>0)

        # Normalize the ROI mask by the number of non-zero pixels
        sum_roi_pixels = np.sum(roi_mask_highlighted)
        if num_non_zero_pixels > 0:
            roi_mask_normalized_highlighted = sum_roi_pixels / num_non_zero_pixels
        else:
            roi_mask_normalized_highlighted = 0  
 
        # cut out ROI from image -> multiply ROI with more weight and then add it to overall image 
        obj_fun_img = obj_fun_sum + gamma * (roi_mask_normalized + roi_mask_normalized_highlighted) 
        obj_fun_img = obj_fun_img + feta * min_distance
        combined_img = alpha * gray + alpha * highlighted_image + beta * roi_mask + gamma * roi_mask_highlighted
        print("Done!")
        return obj_fun_img, combined_img, gray



    def apply_mask(self, image, mask):
        # Use the mask to set pixels outside the ROI to zero
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image

    def calculate_highlights(self, img_masked, img):
        histogram, bin_edges = np.histogram(img_masked, bins=256, range=(0, 256))

        smoothed_histogram = gaussian_filter1d(histogram, sigma=2)
        peaks, _ = find_peaks(smoothed_histogram, distance=10)
        if len(peaks) > 0:
            threshold = peaks[-1]
        else:
            rospy.logerr("ERROR Histogram")
            threshold = 0  # Default to 0 if no peaks are found
        
        max_peak = np.argmax(smoothed_histogram)
        threshold_max = bin_edges[max_peak]
        
        # Find the top 5 peak values in the histogram
        num_peaks = 50
        peaks = np.argpartition(histogram, -num_peaks)[-num_peaks:]
        peaks = np.sort(peaks)  # Sort the peak indices in ascending order

        # Create a mask for pixel values above the threshold
        above_threshold_mask = img > threshold

        # Highlight the values above the threshold in the original image
        highlighted_image = np.ones_like(img) * 255
        highlighted_image[~above_threshold_mask] = 0  # Set all other pixels to 0



        return highlighted_image


    

    def segment_img(self, rgb_img):
        height, width = rgb_img.shape[:2]
        depth_data_copy = np.ones((height, width), dtype=np.uint8)
        segmented_image, segmented_depth, bbox = self.plate_detector.segment_and_detect(rgb_img, depth_data_copy, debug=False)
        return segmented_image, bbox
    
    def vector2array(self, vec):
        return np.array([vec.x,
                  vec.y,
                  vec.z])

    def unpack_vectors(self):
        msg = self.roi
        pt1 = self.vector2array(msg.point1)     
        pt2 = self.vector2array(msg.point2)    
        pt3 = self.vector2array(msg.point3)    
        pt4 = self.vector2array(msg.point4)      

        return pt1, pt2, pt3, pt4
    
    def get_roi(self, current_ee_pose):
        try:
            point_center_edge_1, point_center_edge_2, point_center_edge_1_inner, point_center_edge_2_inner = self.unpack_vectors()
            image_height, image_width, _ = self.rgb_image.shape
            p_img_edge_1 = utils.transform_base_img(point_center_edge_1, current_ee_pose, image_width, image_height)
            p_img_edge_2 = utils.transform_base_img(point_center_edge_2, current_ee_pose, image_width, image_height)
            p_img_inner_1 = utils.transform_base_img(point_center_edge_1_inner, current_ee_pose, image_width, image_height)
            p_img_inner_2 = utils.transform_base_img(point_center_edge_2_inner, current_ee_pose, image_width, image_height)
            roi = np.array([p_img_edge_1[0:2], p_img_edge_2[0:2], p_img_inner_2[0:2], p_img_inner_1[0:2]])
            return roi
        except:
            return None

    def get_ee_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform("base", "tool0_controller", rospy.Time(0), rospy.Duration(1.0))
            return transform.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to get transform: %s", e)

    def numerical_gradient(self, current_point, current_value, epsilon=0.001):
        print(epsilon)
        # Attention: We perform gradient ascent in PC space and not in 7D space
        grad = np.zeros_like(current_point)
        #Actual
        f_point = self.polynomial_model.predict([current_point])
        # Loop over each dimension
        for i in range(len(current_point)):
            point_eps = current_point.copy()
            
            # Perturb the i-th dimension
            point_eps[i] += epsilon
            
            # Compute the function values at the perturbed and original points
            # Estimate
            f_point_eps = self.polynomial_model.predict([point_eps])
            print(f_point_eps)
            print(f_point)
            
            
            # Compute the partial derivative using finite differences
            grad[i] = (f_point_eps - f_point) / epsilon
        
        return grad, f_point


    def fit_model_csv_data(self):
        csv_file = filename = f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/{self.experiment}/record.csv"
        pose_array = []
        obj_array = []

        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                pose_array.append([float(row[1]), float(row[2]), float(row[3]), 
                                float(row[4]), float(row[5]), float(row[6]), float(row[7])])
                obj_array.append(float(row[8]))

        pose_array = np.array(pose_array)
        obj_array = np.array(obj_array)

        self.X = pose_array
        self.Y = obj_array

        pose_array_scaled = self.scaler.fit_transform(pose_array)       
        X_pca = self.PCA.fit_transform(pose_array_scaled)  
        self.polynomial_model = self.polynomial_model.fit(X_pca, obj_array)


##########################################################


    def run(self):
        debug_array = []
        rgb_image_copy = copy.deepcopy(np.asarray(self.rgb_image))
        depth_data_copy = copy.deepcopy(np.asarray(self.depth_image))
        ee_pose = copy.deepcopy(self.get_ee_pose())
        self.last_position = ee_pose
        bboxes = self.plate_detector.detect(rgb_image_copy)
        if bboxes != None:
            bboxes = bboxes.cpu().numpy()

            debug_array.append(str(bboxes[0, 0]))
            debug_array.append(str(bboxes[0, 1]))
            debug_array.append(str(bboxes[0, 2]))
            debug_array.append(str(bboxes[0, 3]))
            # debug_array.append(str(bbox_conf))


             
            try:
                start_segm_timer = time.time()
                rgb_image_copy = copy.deepcopy(np.asarray(self.rgb_image))
                depth_data_copy = copy.deepcopy(np.asarray(self.depth_image))
                rospy.loginfo("Save image and depth")
                cv2.imwrite(self.DEBUG_FOLDER+"rgb_img"+str(self.DEBUG_RUN_NUM)+".jpg", rgb_image_copy)
                np.save(self.DEBUG_FOLDER+"depth_data"+str(self.DEBUG_RUN_NUM)+".npy", depth_data_copy)
                
                ee_pose = copy.deepcopy(self.get_ee_pose())
                segmented_image, segmented_depth = self.plate_detector.segment(rgb_image_copy, depth_data_copy, debug=False)
                eigenvectors, mean, gl_1, gl_2, gw_1, gw_2 = self.grasp_point_estimator.estimate_grasp_point(segmented_image, segmented_depth, self.camera_intrinsics)
                print("ESTIMATED MIDDLE OF THE PLATE")
                print(mean)
                debug_array.append(str(mean[0]))
                debug_array.append(str(mean[1]))
                debug_array.append(str(mean[2]))

                # self.grasp_points.p1_long = gl_1
                # self.grasp_points.p2_long = gl_2
                # self.grasp_points.p1_short = gw_1
                # self.grasp_points.p2_short = gw_2
                # print("Grasping points")
                # print(self.grasp_points)
                
                point_base = utils.transform_pcl_base(mean, ee_pose)
                point_mean = copy.deepcopy(point_base)
                 
                print("In robot base frame we get")
                print(point_base)

                normal_y = utils.rotate_pcl_base(eigenvectors[0, :], ee_pose)
                normal_z = utils.rotate_pcl_base(eigenvectors[2, :], ee_pose)

                if self.MODE == 0:
                    quaternion, point_base, R_tcp = self.grasp_above(normal_z, normal_y, ee_pose, point_base)
                else:
                    quaternion, point_base, R_tcp = self.grasp_side(normal_z, normal_y, ee_pose, point_base)
                point_center_edge_1, point_center_edge_2, point_center_edge_1_inner, point_center_edge_2_inner = utils.get_ocr_roi(point_mean, R_tcp[:, 1], R_tcp[:, 0], self.WIDTH, self.LENGTH)

                self.publish_plate_center(point_center_edge_1, point_center_edge_2, point_center_edge_1_inner, point_center_edge_2_inner)
                recording_state = Bool()
                recording_state.data = True
                self.record_pub.publish(recording_state) 

                print("GRASPING POINT")
                print(point_base)
                print(quaternion)
                # point_base[2] = point_base[2] + 0.35
                # self.position_controller.go_to_pose(point_base, quaternion)
                debug_array.append(str(point_base[0]))
                debug_array.append(str(point_base[1]))
                debug_array.append(str(point_base[2]))

                debug_array.append(str(quaternion[0]))
                debug_array.append(str(quaternion[1]))
                debug_array.append(str(quaternion[2]))
                debug_array.append(str(quaternion[3]))

                ##############################################################
                # Move camera around to record data 
                points, rotated_quats = self.generate_points_sparse(quaternion, point_base, R_tcp[:, 1], normal_z, 0.35)


                # self.position_controller.go_to_pose(point_base, quaternion)

                for i in range(len(points)):
                    print("Point: ", points[i])
                    print("Rotation: ", rotated_quats[i])
                    self.TRACK_OCR = True
        
                    self.position_controller.go_to_pose(points[i],rotated_quats[i])



                # Fit the model on the recorded data
                recording_state.data = False
                self.record_pub.publish(recording_state) 
                self.fit_model_csv_data()
                print(points)
                print(rotated_quats)
                ##############################################################
                self.position_controller.go_to_pose(points[0],rotated_quats[0])
                pred_obj_value = 0
                prev_pred_obj_value = 1000
                output_value = 0
                prev_output_value = 1000
                rospy.loginfo("########## Start Gradient Descent #######")
                while np.abs((prev_pred_obj_value-pred_obj_value)) > 1:
                # while np.abs((prev_output_value-output_value)) > 0.1 and np.abs((prev_pred_obj_value-pred_obj_value)) > 1:
                    print("Difference")
                    print((prev_pred_obj_value-pred_obj_value))
                    learning_rate = 3e-6
                    print("Start fit polynomial model")
                    ee_pose = self.get_ee_pose()
                    print("EE Pose")
                    print(ee_pose)
                    # Update X and Y 
                    rgb_img = copy.deepcopy(self.rgb_image)
                    # PROBLEM (!!!)
                    roi = self.get_roi(ee_pose)
                    try_again = False
                    if roi is not None:
                        print("ROI")
                        print(roi)
                        segmented_image, bbox = self.segment_img(rgb_img)
                        # rospy.logwarn(f"The prediction confidence: {pred_conf}")
                        if segmented_image is not None:
                            output_value, combined_img, gray = self.calc_objective_function(segmented_image, roi, bbox)
                            print("Objective function value")
                            rospy.logwarn(f"The output value is: {output_value}")
                            # Create a rotation object from the quaternion
                            r = Rotation.from_quat([ee_pose.rotation.x,
                                    ee_pose.rotation.y,
                                    ee_pose.rotation.z,
                                    ee_pose.rotation.w])
                            
                            # Convert the rotation to Euler angles
                            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
                            # New data point
                            new_X = np.array([ee_pose.translation.x, 
                                            ee_pose.translation.y, 
                                            ee_pose.translation.z, 
                                            ee_pose.rotation.x,
                                            ee_pose.rotation.y,
                                            ee_pose.rotation.z,
                                            ee_pose.rotation.w])  
                            new_y = output_value
                            print(new_X)
                            print(new_y)

                            self.X = np.vstack([self.X, new_X]) if self.X.size else np.array([new_X])
                            self.Y = np.append(self.Y, new_y)
                            print("Start fit transform")
                            pose_array_scaled = self.scaler.fit_transform(self.X) 
                            print("STart do PCA")      
                            X_pca = self.PCA.fit_transform(pose_array_scaled)  
                            print("Start Polynomial Model")
                            self.polynomial_model = self.polynomial_model.fit(X_pca, self.Y)

                            rospy.loginfo("------------ Done: Fit Polynomial Model ------------")
                            print(X_pca[-1,:])
                            # Compute Gradient 
                            prev_pred_obj_value = copy.deepcopy(pred_obj_value)
                            prev_output_value = copy.deepcopy(output_value)
                            grad, pred_obj_value = self.numerical_gradient(X_pca[-1,:], new_y, epsilon=0.01)
                            print(grad)


                            # Update Step 
                            pca_post = X_pca[-1,:] + learning_rate * grad
                            rospy.loginfo("------------ Done: Update Step Gradient ------------")
                            # Convert back into our 7D space
                            # first PCA
                            pca_post_inverse = self.PCA.inverse_transform(pca_post)
                            print(pca_post_inverse)
                            # then MinMaxScaler
                            pca_post_scaled = self.scaler.inverse_transform(pca_post_inverse.reshape(1, -1))
                            rospy.loginfo("------------ Done: Convert Back to 7D space ------------")
                            
                            pca_post_scaled = pca_post_scaled[0]
                            print("Where are we going?")
                            print(ee_pose)
                            print(pca_post_scaled)

                            point_base = np.array([pca_post_scaled[0],
                                                pca_post_scaled[1],
                                                pca_post_scaled[2]])
                            quaternion = np.array([pca_post_scaled[3],
                                                pca_post_scaled[4],
                                                pca_post_scaled[5],
                                                pca_post_scaled[6]])
                            # Sum of quats always has to be 1
                            quaternion = quaternion / np.linalg.norm(quaternion)
                            # Move there 
                            print(point_base)
                            print(quaternion)
                            # also normalize quaternions 
                            self.position_controller.go_to_pose_gd(point_base, quaternion)

                            self.pose_logging.append(pca_post_scaled)
                            self.write_to_csv(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/{self.experiment}/gradientAscent.csv", [self.iteration_gd ,pca_post, pca_post_scaled, output_value, pred_obj_value])
                            # cv2.imwrite(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/obj_img_{self.iteration_gd}.jpg", combined_img)
                            cv2.imwrite(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/{self.experiment}/gray_img_{self.iteration_gd}.jpg", gray)
                            cv2.imwrite(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/{self.experiment}/img_{self.iteration_gd}.jpg", rgb_img)
                            self.iteration_gd += 1
                        else: 
                            print("no ROI")
                            try_again = True                
                    else:
                        try_again = True
                        rospy.logerr("NO SEGMENTED IMAGE")
                    if try_again == True:
                        print("................... TRY AGAIN .................")
                        output_value = 0
                        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
                        print("Objective function value")
                        rospy.logwarn(f"The output value is: {output_value}")
                        # Create a rotation object from the quaternion
                        r = Rotation.from_quat([ee_pose.rotation.x,
                                ee_pose.rotation.y,
                                ee_pose.rotation.z,
                                ee_pose.rotation.w])
                        
                        # Convert the rotation to Euler angles
                        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
                        # New data point
                        new_X = np.array([ee_pose.translation.x, 
                                        ee_pose.translation.y, 
                                        ee_pose.translation.z, 
                                        ee_pose.rotation.x,
                                        ee_pose.rotation.y,
                                        ee_pose.rotation.z,
                                        ee_pose.rotation.w])  
                        new_y = output_value
                        print(new_X)
                        print(new_y)

                        self.X = np.vstack([self.X, new_X]) if self.X.size else np.array([new_X])
                        self.Y = np.append(self.Y, new_y)
                        print("Start fit transform")
                        pose_array_scaled = self.scaler.fit_transform(self.X) 
                        print("STart do PCA")      
                        X_pca = self.PCA.fit_transform(pose_array_scaled)  
                        print("Start Polynomial Model")
                        self.polynomial_model = self.polynomial_model.fit(X_pca, self.Y)

                        rospy.loginfo("------------ Done: Fit Polynomial Model ------------")
                        print(X_pca[-1,:])
                        # Compute Gradient 
                        prev_pred_obj_value = copy.deepcopy(pred_obj_value)
                        grad, pred_obj_value = self.numerical_gradient(X_pca[-1,:], new_y, epsilon=0.01)
                        print(grad)


                        # Update Step 
                        pca_post = X_pca[-1,:] + learning_rate * grad
                        rospy.loginfo("------------ Done: Update Step Gradient ------------")
                        # Convert back into our 7D space
                        # first PCA
                        pca_post_inverse = self.PCA.inverse_transform(pca_post)
                        print(pca_post_inverse)
                        # then MinMaxScaler
                        pca_post_scaled = self.scaler.inverse_transform(pca_post_inverse.reshape(1, -1))
                        rospy.loginfo("------------ Done: Convert Back to 7D space ------------")
                        
                        pca_post_scaled = pca_post_scaled[0]
                        print("Where are we going?")
                        print(ee_pose)
                        print(pca_post_scaled)

                        point_base = np.array([pca_post_scaled[0],
                                            pca_post_scaled[1],
                                            pca_post_scaled[2]])
                        quaternion = np.array([pca_post_scaled[3],
                                            pca_post_scaled[4],
                                            pca_post_scaled[5],
                                            pca_post_scaled[6]])
                        # Sum of quats always has to be 1
                        quaternion = quaternion / np.linalg.norm(quaternion)
                        # Move there 
                        print(point_base)
                        print(quaternion)
                        # also normalize quaternions 
                        self.position_controller.go_to_pose_gd(point_base, quaternion)

                        self.pose_logging.append(pca_post_scaled)
                        self.write_to_csv(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/{self.experiment}/gradientAscent.csv", [self.iteration_gd ,pca_post, pca_post_scaled, output_value, pred_obj_value])
                        # cv2.imwrite(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/obj_img_{self.iteration_gd}.jpg", combined_img)
                        cv2.imwrite(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/{self.experiment}/gray_img_{self.iteration_gd}.jpg", gray)
                        cv2.imwrite(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/{self.experiment}/img_{self.iteration_gd}.jpg", rgb_img)
                        self.iteration_gd += 1
                print("------------------- DOEN WITH NBV -------------------------")

                return debug_array

            except Exception as e:
                rospy.logerr(f"Error: {str(e)}")

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()

    def normalize_vector(self, vector):
        magnitude = np.linalg.norm(vector)
        if magnitude == 0:
            return vector  # Avoid division by zero
        return vector / magnitude

    def generate_points_extensive(self, quat_tcp, center_point, y_axis, z_axis, radius):
        roll_start = np.radians(0)
        roll_stop = np.radians(25)
        pitch_start = np.radians(-10)
        pitch_stop = np.radians(30)
        yaw_start = np.radians(-20)
        yaw_stop = np.radians(20)

        angle_step = 9
        points = []
        rotated_quats = []

        angle_step_rad = np.radians(angle_step)



        y_axis = self.normalize_vector(y_axis)
        z_axis = np.array([0,0,1])

        x_axis = np.cross(y_axis, z_axis)
        x_axis = self.normalize_vector(x_axis)

        # Calculate the initial point on the circle
        center_point = center_point[:3]
        rotated_quats.append(quat_tcp)

        initial_point = np.array(center_point) - radius * np.sin(roll_start) * y_axis + radius * np.cos(roll_start) * z_axis

        points.append(tuple(initial_point))
        counter = 0

        # Generate points in the circle
        angle = roll_start
        beta_array = np.array([np.radians(-10), np.radians(10)])
        while angle <= roll_stop:
            angle += angle_step_rad
 
            point = np.array(center_point) - radius * np.sin(angle) * y_axis + radius * np.cos(angle) * z_axis
            point[2] -= counter * 0.05

            points.append(tuple(point))   

            r = Rotation.from_quat(quat_tcp)

            # Rotate around the TCP's x-axis by the specified angle
            r_rotated = r * Rotation.from_rotvec([angle, 0, 0])

            # Get the rotated quaternion
            quat_rotated = r_rotated.as_quat()
  
            # Append rotated quaternion to list
            rotated_quats.append(quat_rotated)

            # # HERE GENERATE THE POINTS WHERE WE TURN ALSO AROUND X
            for index, alpha in enumerate(np.array([np.radians(0), np.radians(20)])):
                beta = beta_array[index]
                # point_x = np.array(point) - radius * np.sin(alpha) * x_axis #+ radius * np.cos(alpha) * z_axis
                # point_x[2] -= index * 0.02
                points.append(tuple(point))
                r_x = Rotation.from_quat(quat_rotated)
                r_rotated_x = r_x * Rotation.from_rotvec([0, alpha, beta])
                quat_rotated_x = r_rotated_x.as_quat()
                rotated_quats.append(quat_rotated_x)

            # r_tcp = Rotation.from_quat(quat_tcp)
            counter += 1

        return points, rotated_quats
    
    def generate_points_sparse(self, quat_tcp, center_point, y_axis, z_axis, radius):
        gamma_angle = np.array([np.radians(0), np.radians(20)])

 
        points = []
        rotated_quats = []



        y_axis = self.normalize_vector(y_axis)
        z_axis = np.array([0,0,1])

        x_axis = np.cross(y_axis, z_axis)
        x_axis = self.normalize_vector(x_axis)

        # Calculate the initial point on the circle
        center_point = center_point[:3]
        rotated_quats.append(quat_tcp)

        initial_point = np.array(center_point) - radius * np.sin(gamma_angle[0]) * y_axis + radius * np.cos(gamma_angle[0]) * z_axis

        points.append(tuple(initial_point))
        counter = 0
        # for index, alpha in enumerate(np.array([np.radians(5), np.radians(25)])):
        #         # point_x = np.array(point) - radius * np.sin(alpha) * x_axis #+ radius * np.cos(alpha) * z_axis
        #         # point_x[2] -= index * 0.02
        #         points.append(tuple(initial_point))
        #         r_x = Rotation.from_quat(quat_tcp)
        #         r_rotated_x = r_x * Rotation.from_rotvec([0, alpha, 0])
        #         quat_rotated_x = r_rotated_x.as_quat()
        #         rotated_quats.append(quat_rotated_x)

        # Generate points in the circle

        beta_array = np.array([np.radians(-10), np.radians(10)])
        for i, gamma in enumerate(gamma_angle):
 
            point = np.array(center_point) - radius * np.sin(gamma) * y_axis + radius * np.cos(gamma) * z_axis
            point[2] -= counter * 0.05

            points.append(tuple(point))   

            r = Rotation.from_quat(quat_tcp)

            # Rotate around the TCP's x-axis by the specified angle
            r_rotated = r * Rotation.from_rotvec([gamma, 0, 0])

            # Get the rotated quaternion
            quat_rotated = r_rotated.as_quat()
  
            # Append rotated quaternion to list
            rotated_quats.append(quat_rotated)

            # # HERE GENERATE THE POINTS WHERE WE TURN ALSO AROUND X
            for index, alpha in enumerate(np.array([np.radians(0), np.radians(20)])):
                beta = beta_array[index]
                # point_x = np.array(point) - radius * np.sin(alpha) * x_axis #+ radius * np.cos(alpha) * z_axis
                # point_x[2] -= index * 0.02
                points.append(tuple(point))
                r_x = Rotation.from_quat(quat_rotated)
                r_rotated_x = r_x * Rotation.from_rotvec([0, alpha, beta])
                quat_rotated_x = r_rotated_x.as_quat()
                rotated_quats.append(quat_rotated_x)

            # r_tcp = Rotation.from_quat(quat_tcp)
            counter += 1

        return points, rotated_quats
   


   
    def tf_callback(self, msg):
        try:
            # Loop through all the transforms in the TF message
            for transform in msg.transforms:
                # Check if the transform is the one you are interested in
                if transform.child_frame_id == "marker_24" and transform.header.frame_id == "base":
                    # Update the aruco_position attribute
                    self.aruco_position = transform.transform
                    # rospy.loginfo(f"Board transformation updated: {self.aruco_position}")
        except Exception as e:
            rospy.logwarn(f"Transformation for AruCo Marker: {str(e)}")


    def depth_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.depth_image = cv_image
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {str(e)}")

    def rgb_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb_image = cv_image
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def camera_info_callback(self, msg):
        try:
            self.camera_info = msg.K
            self.camera_intrinsics['f'] = msg.K[0]
            self.camera_intrinsics['cx'] = msg.K[2]
            self.camera_intrinsics['cy'] = msg.K[5]
        except Exception as e:
            rospy.logerr(f"Error getting camera info: {str(e)}")

if __name__ == '__main__':
    print("START")
    rospy.init_node('gradient_descent_pipeline', anonymous=True)
    pipeline = AutomationPipeline()

    # # Spin to keep the script alive
    try:
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            pipeline.start()
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")