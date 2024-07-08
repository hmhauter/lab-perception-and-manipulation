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
from grasp_point_estimator_old import GraspPointEstimator

from ur_master_thesis.msg import GraspPoints
from scipy.spatial.transform import Rotation

from tf2_msgs.msg import TFMessage
import open3d as o3d

class AutomationPipeline():
    def __init__(self):
        # DEBUG VALUES 
        self.DEBUG_RUN_NUM = 35
        self.DEBUG_FOLDER = "/home/apo/catkin_ws/src/ur_master_thesis/src/RESULTS/run10/"
        self.DEBUG_WRITE = []

        UR_IP = "192.168.1.100"
        self.SEGMENT = True
        self.PLATE_ALERT = False

        self.MODE = 0

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

        self.SUCCESS_THRESHOLD = 5
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


        # Create a subscriber
        rospy.Subscriber(camera_info_topic, info_type, self.camera_info_callback)
        rospy.Subscriber(depth_topic, img_type, self.depth_image_callback)
        rospy.Subscriber(rgb_topic, img_type, self.rgb_image_callback)
        
        # fixed measurements
        self.OFFSET_Z_ABOVE = 0.17+0.02 # 0.1776 # m
        self.OFFSET_Y_ABOVE = 0.01525#+0.002  # measured in CAD
        self.OFFSET_Y_SIDE = 0.05 # 0.047 # 0.05 
        self.OFFSET_Z_SIDE = 0.1413  # measured in CAD
        LENGTH = 0.124
        WIDTH = 0.081

        self.plate_detector = PlateDetector("/home/apo/Documents/MasterThesis/adaptive-lab-automation/datasets/runs/segment/train/weights/best.pt")
        self.grasp_point_estimator = GraspPointEstimator(WIDTH, LENGTH)
        self.position_controller = PositionControllerUR()
        
        self.grasp_points = GraspPoints()
        self.grasp_point_pub = rospy.Publisher('/grasp/points',  GraspPoints, queue_size=1)

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
                rospy.loginfo(f"Search Angle: {self.search_angle}")
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
            aruco_position = copy.deepcopy(self.aruco_position)
            aruco_translation, aruco_quat = self.deliver(aruco_position)

            self.position_controller.go_to_pose(aruco_translation, aruco_quat)
            print("ForcE controL")
            rospy.sleep(1.0)
            deliver_flag = False
            deliver_flag = self.position_controller.deliver_plate(gripper=self.gripper)
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
            self.aruco_position = None



    def grasp_above(self, normal_z, normal_y, ee_pose, point_base):
        R_tcp, z_tcp, y_tcp = utils.get_tcp_rot_above(normal_z, normal_y, ee_pose)
        quaternion = Rotation.from_matrix(R_tcp).as_quat()
        point_base[:3] -= self.OFFSET_Z_ABOVE * z_tcp
        point_base[:3] += self.OFFSET_Y_ABOVE * R_tcp[:, 1] 

        return quaternion, point_base

    def grasp_side(self, normal_z, normal_y, ee_pose, point_base):
        R_tcp, z_tcp, y_tcp = utils.get_tcp_rot_side(normal_z, normal_y, ee_pose)
        quaternion = Rotation.from_matrix(R_tcp).as_quat()
        point_base[:3] += self.OFFSET_Z_SIDE * R_tcp[:, 2] 
        point_base[:3] -= self.OFFSET_Y_SIDE * y_tcp

        return quaternion, point_base


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
                cv2.imwrite(self.DEBUG_FOLDER+"segmented_img_"+str(self.DEBUG_RUN_NUM)+".jpg", segmented_image)
                eigenvectors, mean, gl_1, gl_2, gw_1, gw_2, point_cloud, line_set, pcl, mesh_coordinate_frame, registered_pointcloud, fitness, pcl_surface, pcl_surface_smooth = self.grasp_point_estimator.estimate_grasp_point(segmented_image, segmented_depth, self.camera_intrinsics)
                
                o3d.io.write_point_cloud(self.DEBUG_FOLDER+"point_cloud_"+str(self.DEBUG_RUN_NUM)+".ply", point_cloud)
                o3d.io.write_point_cloud(self.DEBUG_FOLDER+"pcl_"+str(self.DEBUG_RUN_NUM)+".ply", pcl),
                o3d.io.write_point_cloud(self.DEBUG_FOLDER+"surface_"+str(self.DEBUG_RUN_NUM)+".ply", pcl_surface),
                o3d.io.write_point_cloud(self.DEBUG_FOLDER+"registered_"+str(self.DEBUG_RUN_NUM)+".ply", registered_pointcloud)
                o3d.io.write_point_cloud(self.DEBUG_FOLDER+"smooth_"+str(self.DEBUG_RUN_NUM)+".ply", pcl_surface_smooth)
                o3d.io.write_triangle_mesh(self.DEBUG_FOLDER+"mesh_coordinate_frame_"+str(self.DEBUG_RUN_NUM)+".ply", mesh_coordinate_frame)
                o3d.io.write_line_set(self.DEBUG_FOLDER+"line_set_"+str(self.DEBUG_RUN_NUM)+".ply", line_set)


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

                normal_y = utils.rotate_pcl_base(eigenvectors[0, :], ee_pose)
                normal_z = utils.rotate_pcl_base(eigenvectors[2, :], ee_pose)

                if self.MODE == 0:
                    quaternion, point_base = self.grasp_above(normal_z, normal_y, ee_pose, point_base)
                else:
                    quaternion, point_base = self.grasp_side(normal_z, normal_y, ee_pose, point_base)

                self.position_controller.go_to_pose(point_base, quaternion)

                debug_array.append(self.aruco_position.translation.x)
                debug_array.append(self.aruco_position.translation.y)
                debug_array.append(self.aruco_position.translation.z)

                debug_array.append(self.aruco_position.rotation.x)
                debug_array.append(self.aruco_position.rotation.y)
                debug_array.append(self.aruco_position.rotation.z)
                debug_array.append(self.aruco_position.rotation.w)


                debug_array.append(str(point_base[0]))
                debug_array.append(str(point_base[1]))
                debug_array.append(str(point_base[2]))

                debug_array.append(str(quaternion[0]))
                debug_array.append(str(quaternion[1]))
                debug_array.append(str(quaternion[2]))
                debug_array.append(str(quaternion[3]))

                self.DEBUG_WRITE.append(str(fitness))

                self.gripper.move_and_wait_for_pos(72, 10, 42)
                debug_array.append(start_segm_timer-self.start_time)
                
                self.PICK_SUCCESS = 1

                # self.position_controller.attach_collision_object()

                rospy.loginfo(f"Go back to EE position: {ee_pose}")
                self.position_controller.go_back()
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

                    # Update the aruco_position attribute
                    self.aruco_position = transform.transform
                    rospy.loginfo(f"Board transformation updated: {self.aruco_position}")
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
