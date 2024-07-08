#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import copy
import tf2_ros
import open3d as o3d
import time
from matplotlib import pyplot as plt
from position_controller import PositionControllerUR
from moxa import MoxaRelay
import utils
import transforms3d.quaternions as quat
import robotiq_gripper
import csv
from plate_detector import PlateDetector
from grasp_point_estimator_old import GraspPointEstimator
from geometry_msgs.msg import Point
from std_msgs.msg import String
from ur_master_thesis.msg import GraspPoints, PlateROI 
from scipy.spatial.transform import Rotation

from tf2_msgs.msg import TFMessage

class AutomationPipeline():
    def __init__(self):
        # DEBUG VALUES 
        self.DEBUG_RUN_NUM = 100
        self.DEBUG_FOLDER = "/home/apo/catkin_ws/src/ur_master_thesis/src/RESULTS/run11/"
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
        self.SEARCH_MAX_BASE = 40

        self.SEARCH_MIN_WRIST = -20
        self.SEARCH_MAX_WRIST = 20

        self.ANGLE_INCREMENT = 3
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
        self.has_chars = False
        self.char_counter = 0
        self.HAS_CHARS = False
        self.HAS_CHARS_SNAPHOT = False

        self.has_chars2 = False
        self.char_counter2 = 0
        self.HAS_CHARS2 = False
        self.HAS_CHARS2_SNAPHOT = False

        self.TURN_ME = False

        self.moxa = MoxaRelay()


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
        self.center_point_pub2 = rospy.Publisher('/plate/center2', PlateROI, queue_size=10)

        self.start_ocr_pub = rospy.Publisher('/ocr/start', String, queue_size=1)

        # Create a subscriber
        rospy.Subscriber(camera_info_topic, info_type, self.camera_info_callback)
        rospy.Subscriber(depth_topic, img_type, self.depth_image_callback)
        rospy.Subscriber(rgb_topic, img_type, self.rgb_image_callback)
        
        # fixed measurements
        self.OFFSET_Z_ABOVE = 0.17 + 0.02 # 0.1776 # m
        self.OFFSET_Y_ABOVE = 0.01525#+0.002  # measured in CAD
        self.OFFSET_Y_SIDE = 0.05 # 0.047 # 0.05 
        self.OFFSET_Z_SIDE = 0.1413  # measured in CAD
        LENGTH = 0.124
        WIDTH = 0.081

        self.plate_detector = PlateDetector("/home/apo/Documents/MasterThesis/adaptive-lab-automation/datasets/runs/segment/train/weights/best.pt")
        self.grasp_point_estimator = GraspPointEstimator(WIDTH, LENGTH)
        self.position_controller = PositionControllerUR()

        rospy.Subscriber('detected_text', String, self.detected_ocr_text)
        rospy.Subscriber('detected_text_2', String, self.detected_ocr_text2)
        
        # self.grasp_points = GraspPoints()
        # self.grasp_point_pub = rospy.Publisher('/grasp/points',  GraspPoints, queue_size=1)

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
                    board_transformation.translation.z + self.OFFSET_Z_ABOVE + 0.035
                ])
            else:
            # Define the rotation angle (+90 degrees around local X-axis)
                x_rotation_angle = 90  # Degrees
                final_translation = np.array([
                    board_transformation.translation.x,
                    board_transformation.translation.y + self.OFFSET_Z_SIDE,
                    board_transformation.translation.z + self.OFFSET_Y_SIDE + 0.035
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
        print("&&&&&&&&&&&&&&start&&&&&&&&&&&&&&&&&&&&&&6")
        print(self.HAS_CHARS)
        print(self.HAS_CHARS2)
        print(self.HAS_CHARS_SNAPHOT)
        print(self.HAS_CHARS2_SNAPHOT)
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

                bbox = self.plate_detector.detect(self.rgb_image)
                rospy.logerr(f"Serach Angle: {self.search_angle}")
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
            print("&&&&&&&&&&&&&delivermode&&&&&&&&&&&&&&&&&&&&&&&6")
            print(self.HAS_CHARS)
            print(self.HAS_CHARS2)
            print(self.HAS_CHARS_SNAPHOT)
            print(self.HAS_CHARS2_SNAPHOT)
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
            self.moxa.setLightOff()
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&6")
            print(self.HAS_CHARS)
            print(self.HAS_CHARS2)
            print(self.HAS_CHARS_SNAPHOT)
            print(self.HAS_CHARS2_SNAPHOT)
            if self.HAS_CHARS == False and self.TURN_ME == False:
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                self.position_controller.go_to_pose_intermediate(aruco_translation, aruco_quat)

            self.position_controller.go_to_pose(aruco_translation, aruco_quat)
            self.TURN_ME = False
            print("ForcE controL")
            rospy.sleep(1.0)
            deliver_flag = False
            deliver_flag = self.position_controller.deliver_plate(self.gripper)
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
            print("Reset values... ")
            self.HAS_CHARS = False  
            self.HAS_CHARS2 = False 
            self.HAS_CHARS_SNAPHOT = False  
            self.HAS_CHARS2_SNAPHOT = False 
            self.TURN_ME = False



    def grasp_above(self, normal_z, normal_y, ee_pose, point_base):
        R_tcp, z_tcp, y_tcp = utils.get_tcp_rot_above(normal_z, normal_y, ee_pose)
        quaternion = Rotation.from_matrix(R_tcp).as_quat()
        point_base[:3] -= self.OFFSET_Z_ABOVE * z_tcp
        point_base[:3] += self.OFFSET_Y_ABOVE * R_tcp[:, 1] 

        print("R TCP")
        print(R_tcp)

        print(point_base)

        point_base_rot = copy.deepcopy(point_base)
        point_base_rot[:3] -= (self.OFFSET_Y_ABOVE+0.015)* R_tcp[:, 1] 

        return quaternion, point_base, R_tcp, point_base_rot

    def grasp_side(self, normal_z, normal_y, ee_pose, point_base):
        R_tcp, z_tcp, y_tcp = utils.get_tcp_rot_side(normal_z, normal_y, ee_pose)
        quaternion = Rotation.from_matrix(R_tcp).as_quat()
        point_base[:3] += self.OFFSET_Z_SIDE * R_tcp[:, 2] 
        point_base[:3] -= self.OFFSET_Y_SIDE * y_tcp

        point_base_rot = copy.deepcopy(point_base)

        return quaternion, point_base, R_tcp, point_base_rot

    def pick_up_plate_for_light(self):
        self.gripper.move_and_wait_for_pos(0, 10, 10)
        if self.HAS_CHARS == False and self.TURN_ME == False:
            point_mean, y_axis, x_axis = self.position_controller.deliver_go_to_light_position_rot(self.gripper)
        else:
            point_mean, y_axis, x_axis = self.position_controller.deliver_go_to_light_position(self.gripper)
        self.stop_ocr()
     

    def deliver_plate_for_light(self):
        point_mean, y_axis, x_axis = self.position_controller.go_to_light_position()
        self.gripper.move_and_wait_for_pos(0, 10, 10)
        rospy.sleep(0.3)
        
        point_center_edge_1, point_center_edge_2, point_center_edge_1_inner, point_center_edge_2_inner = utils.get_ocr_roi(point_mean, y_axis, x_axis, self.WIDTH, self.LENGTH)
        self.start_ocr()
        self.publish_plate_center(point_center_edge_1, point_center_edge_2, point_center_edge_1_inner, point_center_edge_2_inner)

        self.position_controller.go_to_detection_position()
        rospy.sleep(3)

    def create_vec(self, point):
        vec = Point()
        vec.x = point[0]
        vec.y = point[1]
        
        vec.z = point[2]
        return vec
    
    def start_ocr(self):
        msg = String()
        msg.data = str(True)
        self.start_ocr_pub.publish(msg)

    def stop_ocr(self):
        msg = String()
        msg.data = str(False)
        self.start_ocr_pub.publish(msg)
    
    def publish_plate_center(self, pt1, pt2, pt3, pt4):
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
        print("(((((((((((((((((())))))))))))))))))8((((((((((((()))))))))))))")
        print(plate_roi)

        self.center_point_pub.publish(plate_roi)
    def publish_plate_center2(self, pt1, pt2, pt3, pt4):
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

        self.center_point_pub2.publish(plate_roi)
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

             
            try:
                start_segm_timer = time.time()
                rgb_image_copy = copy.deepcopy(np.asarray(self.rgb_image))
                depth_data_copy = copy.deepcopy(np.asarray(self.depth_image))
                rospy.loginfo("Save image and depth")
                cv2.imwrite(self.DEBUG_FOLDER+"rgb_img"+str(self.DEBUG_RUN_NUM)+".jpg", rgb_image_copy)
                np.save(self.DEBUG_FOLDER+"depth_data"+str(self.DEBUG_RUN_NUM)+".npy", depth_data_copy)
                
                ee_pose = copy.deepcopy(self.get_ee_pose())

                segmented_image, segmented_depth = self.plate_detector.segment(rgb_image_copy, depth_data_copy, debug=False)
        
                eigenvectors, mean, gl_1, gl_2, gw_1, gw_2, point_cloud, line_set, pcl, mesh_coordinate_frame, registered_pointcloud, fitness, pcl_surface, pcl_surface_smooth  = self.grasp_point_estimator.estimate_grasp_point(segmented_image, segmented_depth, self.camera_intrinsics)
          
                cv2.imwrite(self.DEBUG_FOLDER+"segmented_img_"+str(self.DEBUG_RUN_NUM)+".jpg", segmented_image)
                print("Write all the pointclouds...")
                o3d.io.write_point_cloud(self.DEBUG_FOLDER+"point_cloud_"+str(self.DEBUG_RUN_NUM)+".ply", point_cloud)
                o3d.io.write_point_cloud(self.DEBUG_FOLDER+"pcl_"+str(self.DEBUG_RUN_NUM)+".ply", pcl),
                o3d.io.write_point_cloud(self.DEBUG_FOLDER+"surface_"+str(self.DEBUG_RUN_NUM)+".ply", pcl_surface),
                o3d.io.write_point_cloud(self.DEBUG_FOLDER+"registered_"+str(self.DEBUG_RUN_NUM)+".ply", registered_pointcloud)
                o3d.io.write_point_cloud(self.DEBUG_FOLDER+"smooth_"+str(self.DEBUG_RUN_NUM)+".ply", pcl_surface_smooth)
                o3d.io.write_triangle_mesh(self.DEBUG_FOLDER+"mesh_coordinate_frame_"+str(self.DEBUG_RUN_NUM)+".ply", mesh_coordinate_frame)
                o3d.io.write_line_set(self.DEBUG_FOLDER+"line_set_"+str(self.DEBUG_RUN_NUM)+".ply", line_set)

                self.DEBUG_WRITE.append(str(fitness))
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
                print("In robot base frame we get")
                print(point_base)
                point_mean = copy.deepcopy(point_base)

                normal_y = utils.rotate_pcl_base(eigenvectors[0, :], ee_pose)
                normal_z = utils.rotate_pcl_base(eigenvectors[2, :], ee_pose)

                if self.MODE == 0:
                    quaternion, point_base, R_tcp, point_base_rot = self.grasp_above(normal_z, normal_y, ee_pose, point_base)
                else:
                    quaternion, point_base, R_tcp, point_base_rot = self.grasp_side(normal_z, normal_y, ee_pose, point_base)
                print("GRASPING POINT")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(point_base)
                print(quaternion)

                self.moxa.setLightOn()
                self.start_ocr()
                point_center_edge_1, point_center_edge_2, point_center_edge_1_inner, point_center_edge_2_inner = utils.get_ocr_roi(point_mean, R_tcp[:, 1], R_tcp[:, 0], self.WIDTH, self.LENGTH)
                point_center_edge_21, point_center_edge_22, point_center_edge_21_inner, point_center_edge_22_inner = utils.get_ocr_roi_2(point_mean, R_tcp[:, 1], R_tcp[:, 0], self.WIDTH, self.LENGTH)
        
                self.publish_plate_center(point_center_edge_1, point_center_edge_2, point_center_edge_1_inner, point_center_edge_2_inner)
                self.publish_plate_center2(point_center_edge_21, point_center_edge_22, point_center_edge_21_inner, point_center_edge_22_inner)
             
                point_base[2] += 0.14
                self.position_controller.go_to_pose(point_base, quaternion)
                point_base[2] -= 0.14
                self.stop_ocr()
                if self.HAS_CHARS2 == True:
                    self.TURN_ME = True
                    qx, qy, qz, qw = quaternion
                    quaternion = np.array([
                        -qy,
                        qx,
                        -qw,
                        qz
                    ])
                    point_base = point_base_rot
                self.position_controller.go_to_pose(point_base, quaternion)
                
                debug_array.append(str(point_base[0]))
                debug_array.append(str(point_base[1]))
                debug_array.append(str(point_base[2]))

                debug_array.append(str(quaternion[0]))
                debug_array.append(str(quaternion[1]))
                debug_array.append(str(quaternion[2]))
                debug_array.append(str(quaternion[3]))

                self.gripper.move_and_wait_for_pos(70, 10, 25)
                debug_array.append(start_segm_timer-self.start_time)
                
                self.PICK_SUCCESS = 1

                # self.position_controller.attach_collision_object()

                rospy.loginfo(f"Go back to EE position: {ee_pose}")
                self.position_controller.go_back()
                print("WHER SHOULD WE GO??")
                
                self.HAS_CHARS_SNAPHOT = copy.deepcopy(self.HAS_CHARS)
                self.HAS_CHARS2_SNAPHOT = copy.deepcopy(self.HAS_CHARS2)
                print("///////////////////////////////")
                print(self.HAS_CHARS)
                print(self.HAS_CHARS2)
                if self.HAS_CHARS == False and self.HAS_CHARS2 == False:
                    self.deliver_plate_for_light()
                    self.pick_up_plate_for_light()


                return debug_array
            except Exception as e:
                rospy.logerr(f"Error: {str(e)}")

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
   
    def tf_callback(self, msg):
        try:
            # Loop through all the transforms in the TF message
            for transform in msg.transforms:
                # Check if the transform is the one you are interested in
                if transform.child_frame_id == "marker_24" and transform.header.frame_id == "base":
                    transform.transform.translation.x = -0.3005297057548497
                    transform.transform.translation.y = -0.5495974321616232
                    transform.transform.translation.z = -0.00442342318402561 + 0.012
                
                    transform.transform.rotation.x = 0.0028161101993548748
                    transform.transform.rotation.y = 0.005343460172666312
                    transform.transform.rotation.z = 0.9999754831723403
                    transform.transform.rotation.w = 0.0035425994654611974
                    self.aruco_position = transform.transform
                    # rospy.loginfo(f"Board transformation updated: {self.aruco_position}")
        except Exception as e:
            rospy.logwarn(f"Transformation for AruCo Marker: {str(e)}")

    def detected_ocr_text2(self, msg):
        try:
            # FALSE means has chars and TRUE menas is empty !
            # yes that happens if there is more than a month between implementations...
            self.has_chars2 = msg.data

            if self.has_chars2 == "False":
                self.char_counter2 += 1
                if self.char_counter2 >= 7:
                    self.HAS_CHARS2 = True
            else:
                self.char_counter2 = 0
            print("-----DETECTD OCR TEXT2----------")
            print(type(self.has_chars2))
            print(msg.data)
            print(self.char_counter2)
            print("Has chars 2?")
            print(self.HAS_CHARS2)
        except Exception as e:
            rospy.logerr(f"Error processing OCR: {str(e)}")

    def detected_ocr_text(self, msg):
        try:
            # FALSE means has chars and TRUE menas is empty !
            self.has_chars = msg.data

            if self.has_chars == "False":
                self.char_counter += 1
                if self.char_counter >= 7:
                    self.HAS_CHARS = True
            else:
                self.char_counter -= 1
                if self.char_counter <= 0:
                    self.char_counter = 0
            print("-----DETECTD OCR TEXT----------")
            print(type(self.has_chars))
            print(msg.data)
            print(self.char_counter)
            print("Has chars?")
            print(self.HAS_CHARS)
        except Exception as e:
            rospy.logerr(f"Error processing OCR: {str(e)}")
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
    rospy.init_node('automation_pipeline', anonymous=True)
    pipeline = AutomationPipeline()

    # # Spin to keep the script alive
    try:
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            pipeline.start()
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")
