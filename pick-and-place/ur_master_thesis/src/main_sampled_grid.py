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
from moxa import MoxaRelay
import utils
import transforms3d.quaternions as quat
import robotiq_gripper
import csv
from plate_detector import PlateDetector
from grasp_point_estimator import GraspPointEstimator
from std_msgs.msg import String, Float32
from ur_master_thesis.msg import GraspPoints, PlateROI
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Point

from ocr_roi import OCR_ROI

from tf2_msgs.msg import TFMessage


class AutomationPipeline:
    def __init__(self):
        # DEBUG VALUES
        self.DEBUG_RUN_NUM = 0
        self.DEBUG_FOLDER = "/test1/"
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

        self.SEARCH_MIN_BASE = -180
        self.SEARCH_MAX_BASE = 180

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
        self.camera_position = None

        self.TRACK_OCR = False
        self.OCR_DONE = False
        self.ocr_counter = 0
        self.ocr_detection = []

        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        rgb_topic = "/camera/color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        img_type = Image
        info_type = CameraInfo

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber("/tf", TFMessage, self.tf_callback)

        rospy.Subscriber("/detected_text", String, self.ocr_callback)
        rospy.Subscriber("/correction_angle", Float32, self.correction_angle_callback)

        rospy.Subscriber(camera_info_topic, info_type, self.camera_info_callback)
        rospy.Subscriber(depth_topic, img_type, self.depth_image_callback)
        rospy.Subscriber(rgb_topic, img_type, self.rgb_image_callback)

        # measured in CAD
        self.OFFSET_Z_ABOVE = 0.17 + 0.025
        self.OFFSET_Y_ABOVE = -0.1
        self.OFFSET_Y_SIDE = 0.05
        self.OFFSET_Z_SIDE = 0.1413
        LENGTH = 0.124
        WIDTH = 0.081

        self.plate_detector = PlateDetector("object-detection/model/best.pt")
        self.grasp_point_estimator = GraspPointEstimator(self.WIDTH, self.LENGTH)
        self.position_controller = PositionControllerUR()

        self.grasp_points = GraspPoints()
        self.grasp_point_pub = rospy.Publisher(
            "/grasp/points", GraspPoints, queue_size=1
        )

        self.center_point_pub = rospy.Publisher(
            "/plate/center", PlateROI, queue_size=10
        )

        self.ocrROI = OCR_ROI()
        self.moxa = MoxaRelay()

    def write_to_csv(self, filename, data):

        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)

            writer.writerow(data)

    def create_to_csv(self, filename, data, headers=None):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            if headers:
                writer.writerow(headers)
            writer.writerows(data)

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

    def plot_points_and_tcp(self, points, quat_tcp, plate):
        from scipy.spatial.transform import Rotation as R

        points = np.array(points)
        quat_tcp = np.array(quat_tcp)

        colors = {
            0: (231.999 / 255, 63.010 / 255, 72.012 / 255),
            1: (0.000, 135.992 / 255, 52.989 / 255),
            2: (251.991 / 255, 117.989 / 255, 51.995 / 255),
            3: (120.997 / 255, 35.011 / 255, 142.009 / 255),
        }
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors[0],
            marker="o",
            label="Camera Points",
        )
        ax.scatter(
            plate[0],
            plate[1],
            plate[2],
            c=colors[2],
            marker="*",
            s=100,
            label="Plate Center",
        )

        tcp_label_added = False
        for point, quat in zip(points, quat_tcp):

            rotation = R.from_quat(quat)
            rotation_matrix = rotation.as_matrix()

            z_axis = rotation_matrix[:, 2]

            if not tcp_label_added:
                ax.quiver(
                    point[0],
                    point[1],
                    point[2],
                    z_axis[0],
                    z_axis[1],
                    z_axis[2],
                    length=0.08,
                    color=colors[1],
                    label="TCP z-axis",
                )
                tcp_label_added = True
            else:
                ax.quiver(
                    point[0],
                    point[1],
                    point[2],
                    z_axis[0],
                    z_axis[1],
                    z_axis[2],
                    length=0.08,
                    color=colors[1],
                )

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("3D Points and TCP z-axes")

        ax.legend()

        plt.show()

    def normalize_vector(self, vector):
        magnitude = np.linalg.norm(vector)
        if magnitude == 0:
            return vector
        return vector / magnitude

    from scipy.spatial.transform import Rotation

    def generate_points(self, quat_tcp, center_point, y_axis, z_axis, radius):

        start_angle = 0
        end_angle = 25
        angle_step = 5
        points = []
        rotated_quats = []

        start_angle_rad = np.radians(start_angle)
        end_angle_rad = np.radians(end_angle)
        angle_step_rad = np.radians(angle_step)

        y_axis = self.normalize_vector(y_axis)
        z_axis = np.array([0, 0, 1])

        x_axis = np.cross(y_axis, z_axis)
        x_axis = self.normalize_vector(x_axis)

        center_point = center_point[:3]
        rotated_quats.append(quat_tcp)

        initial_point = (
            np.array(center_point)
            - radius * np.sin(start_angle_rad) * y_axis
            + radius * np.cos(start_angle_rad) * z_axis
        )

        points.append(tuple(initial_point))
        counter = 0

        angle = start_angle_rad
        while angle <= end_angle_rad:
            angle += angle_step_rad

            point = (
                np.array(center_point)
                - 2 * radius * np.sin(angle) * y_axis
                + radius * np.cos(angle) * z_axis
            )

            point[2] -= counter * 0.01

            points.append(tuple(point))

            r = Rotation.from_quat(quat_tcp)

            r_rotated = r * Rotation.from_rotvec([angle, 0, 0])

            quat_rotated = r_rotated.as_quat()

            rotated_quats.append(quat_rotated)

            for index, alpha in enumerate(
                np.array(
                    [
                        np.radians(5),
                        np.radians(10),
                        np.radians(15),
                        np.radians(20),
                        np.radians(25),
                    ]
                )
            ):
                point_x = np.array(point) - radius * np.sin(alpha) * x_axis
                point_x[2] -= index * 0.02
                points.append(tuple(point_x))
                r_x = Rotation.from_quat(quat_rotated)
                r_rotated_x = r_x * Rotation.from_rotvec([0, alpha, 0])
                quat_rotated_x = r_rotated_x.as_quat()
                rotated_quats.append(quat_rotated_x)

            counter += 1

        return points, rotated_quats

    def generate_points_plate(self, quat_tcp, center_point, y_axis, z_axis):
        points = []
        rotated_quats = []

        step_size = 0.005

        y_axis = self.normalize_vector(y_axis)
        z_axis = np.array([0, 0, 1])

        x_axis = np.cross(y_axis, z_axis)
        x_axis = self.normalize_vector(x_axis)

        center_point = center_point[:3]
        rotated_quats.append(quat_tcp)

        initial_point = np.array(center_point) + 0.05 * y_axis + 0.04 * x_axis

        points.append(tuple(initial_point))
        counter = 0

        angle = 0
        while angle <= 0.14:
            angle += step_size

            point = np.array(initial_point) - angle * y_axis

            points.append(tuple(point))

            rotated_quats.append(quat_tcp)

            counter += 1
        return points, rotated_quats

    def deliver(self, board_transformation):
        ee_pose = self.get_ee_pose()
        ee_quat = np.array(
            [
                ee_pose.rotation.x,
                ee_pose.rotation.y,
                ee_pose.rotation.z,
                ee_pose.rotation.w,
            ]
        )

        original_quat = np.array(
            [
                board_transformation.rotation.x,
                board_transformation.rotation.y,
                board_transformation.rotation.z,
                board_transformation.rotation.w,
            ]
        )

        original_rot_matrix = Rotation.from_quat(original_quat).as_matrix()
        if self.MODE == 0:
            x_rotation_angle = 180
            final_translation = np.array(
                [
                    board_transformation.translation.x,
                    board_transformation.translation.y,
                    board_transformation.translation.z + self.OFFSET_Z_ABOVE + 0.01,
                ]
            )
        else:

            x_rotation_angle = 90
            final_translation = np.array(
                [
                    board_transformation.translation.x,
                    board_transformation.translation.y + self.OFFSET_Z_SIDE,
                    board_transformation.translation.z + self.OFFSET_Y_SIDE + 0.02,
                ]
            )

        x_rotation_matrix = Rotation.from_euler(
            "x", x_rotation_angle, degrees=True
        ).as_matrix()

        y_rotation_angle = 180
        y_rotation_matrix = Rotation.from_euler(
            "z", y_rotation_angle, degrees=True
        ).as_matrix()

        x_new_rot_matrix = np.dot(original_rot_matrix, x_rotation_matrix)
        xy_new_rot_matrix = np.dot(x_new_rot_matrix, y_rotation_matrix)

        x_new_quat = Rotation.from_matrix(x_new_rot_matrix).as_quat()
        xy_new_quat = Rotation.from_matrix(xy_new_rot_matrix).as_quat()

        q_delta1 = x_new_quat * ee_quat.conj()
        q_delta2 = xy_new_quat * ee_quat.conj()

        angles1 = Rotation.from_quat(q_delta1).as_euler("xyz")
        angles2 = Rotation.from_quat(q_delta2).as_euler("xyz")
        angle1 = np.linalg.norm(angles1)
        angle2 = np.linalg.norm(angles2)

        if angle1 < angle2:

            final_quat = x_new_quat
        else:

            final_quat = xy_new_quat

        return final_translation, final_quat

    def start(self):
        rospy.loginfo("Start main script...")
        if self.PICK_SUCCESS == 0:

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
            while (
                bbox == None
                or detection_success <= self.SUCCESS_THRESHOLD
                or is_inside == False
            ):
                rospy.loginfo(f"Search Angle: {self.search_angle}")
                bbox = self.plate_detector.detect(self.rgb_image)
                if bbox != None:
                    detection_success += 1
                    rospy.loginfo(f"Bounding Box:  {bbox}")
                    bboxes = bbox.cpu().numpy()
                    is_inside, side, x_direction, y_direction, vertical_direction = (
                        utils.analyze_bounding_box(bboxes[0])
                    )
                    if (
                        is_inside == False
                        and detection_success >= self.SUCCESS_THRESHOLD
                    ):
                        if x_direction == "too much" and side == "right":
                            self.direction = -1
                            self.position_controller.move_base_joint(
                                self.ANGLE_INCREMENT * self.direction
                            )
                        if x_direction == "too much" and side == "left":
                            self.direction = 1
                            self.position_controller.move_base_joint(
                                self.ANGLE_INCREMENT * self.direction
                            )
                        if y_direction == "too much" and vertical_direction == "up":
                            self.position_controller.move_wrist1_joint(
                                1 * self.ANGLE_INCREMENT_WRIST
                            )
                        if y_direction == "too much" and vertical_direction == "down":
                            self.position_controller.move_wrist1_joint(
                                -1 * self.ANGLE_INCREMENT_WRIST
                            )
                    else:
                        self.position_controller.move_base_joint(
                            self.ANGLE_INCREMENT * self.direction
                        )

                else:
                    if detection_success > 0:
                        detection_success -= 1

                    if (
                        self.search_angle >= self.SEARCH_MAX_BASE
                        or self.search_angle <= self.SEARCH_MIN_BASE
                    ):
                        # Change direction
                        self.direction *= -1
                        if self.direction == -1:
                            # MOVE WRIST
                            if (
                                self.search_angle_wrist >= self.SEARCH_MAX_WRIST
                                or self.search_angle_wrist <= self.SEARCH_MIN_WRIST
                            ) and is_inside == False:
                                # Change direction
                                self.direction_wrist *= -1
                            self.position_controller.move_wrist1_joint(
                                self.ANGLE_INCREMENT_WRIST * self.direction_wrist
                            )
                            self.search_angle_wrist += (
                                self.ANGLE_INCREMENT_WRIST * self.direction_wrist
                            )
                    self.search_angle += self.ANGLE_INCREMENT * self.direction
                    self.position_controller.move_base_joint(
                        self.ANGLE_INCREMENT * self.direction
                    )
                    angle_counter += self.ANGLE_INCREMENT

            debug_array = self.run()
            self.DEBUG_WRITE.extend(debug_array)

        elif self.PICK_SUCCESS == 1:

            rospy.loginfo("Deliver Mode")

            while self.aruco_position == None:

                self.position_controller.move_base_joint(
                    self.ANGLE_INCREMENT * self.direction
                )

                if (
                    self.search_angle >= self.SEARCH_MAX
                    or self.search_angle <= self.SEARCH_MIN
                ):

                    self.direction *= -1
                self.search_angle += self.ANGLE_INCREMENT * self.direction

            deliver_timer = time.time()
            aruco_translation, aruco_quat = self.deliver(self.aruco_position)

            self.position_controller.go_to_pose(aruco_translation, aruco_quat)

            rospy.sleep(1.0)
            deliver_flag = False
            deliver_flag = self.position_controller.deliver_plate()
            if deliver_flag:
                self.gripper.move_and_wait_for_pos(0, 10, 10)
                self.PICK_SUCCESS = 0

            stop_time = time.time()
            self.DEBUG_WRITE.append(str(stop_time - deliver_timer))
            self.DEBUG_WRITE.append(str(stop_time - self.start_time))

            self.write_to_csv(self.DEBUG_FOLDER + "test.csv", self.DEBUG_WRITE)
            self.DEBUG_WRITE = []

    def grasp_above(self, normal_z, normal_y, ee_pose, point_base):
        R_tcp, z_tcp, y_tcp = utils.get_tcp_rot_above(normal_z, normal_y, ee_pose)
        quaternion = Rotation.from_matrix(R_tcp).as_quat()
        point_base[:3] -= self.OFFSET_Z_ABOVE * z_tcp
        point_base[:3] += self.OFFSET_Y_ABOVE * R_tcp[:, 1]

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

        self.center_point_pub.publish(plate_roi)

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
                cv2.imwrite(
                    self.DEBUG_FOLDER + "rgb_img" + str(self.DEBUG_RUN_NUM) + ".jpg",
                    rgb_image_copy,
                )
                np.save(
                    self.DEBUG_FOLDER + "depth_data" + str(self.DEBUG_RUN_NUM) + ".npy",
                    depth_data_copy,
                )

                ee_pose = copy.deepcopy(self.get_ee_pose())
                segmented_image, segmented_depth = self.plate_detector.segment(
                    rgb_image_copy, depth_data_copy, debug=False
                )

                box_angle, box_unit_vector = self.ocrROI.getAngle(segmented_image)
                box_vector_base = utils.angle_camera_base(box_unit_vector, ee_pose)

                (
                    eigenvectors,
                    mean,
                    gl_1,
                    gl_2,
                    gw_1,
                    gw_2,
                    point_cloud,
                    line_set,
                    pcl,
                    mesh_coordinate_frame,
                    registered_pointcloud,
                    fitness,
                    pcl_surface,
                    pcl_surface_smooth,
                ) = self.grasp_point_estimator.estimate_grasp_point(
                    segmented_image, segmented_depth, self.camera_intrinsics
                )

                debug_array.append(str(mean[0]))
                debug_array.append(str(mean[1]))
                debug_array.append(str(mean[2]))

                point_base = utils.transform_pcl_base(mean, ee_pose)

                point_mean = copy.deepcopy(point_base)

                normal_y = utils.rotate_pcl_base(eigenvectors[0, :], ee_pose)
                normal_z = utils.rotate_pcl_base(eigenvectors[2, :], ee_pose)

                image_height, image_width, _ = rgb_image_copy.shape
                if self.MODE == 0:
                    quaternion, point_base, R_tcp = self.grasp_above(
                        normal_z, normal_y, ee_pose, point_base
                    )
                else:
                    quaternion, point_base, R_tcp = self.grasp_side(
                        normal_z, normal_y, ee_pose, point_base
                    )

                (
                    point_center_edge_1,
                    point_center_edge_2,
                    point_center_edge_1_inner,
                    point_center_edge_2_inner,
                ) = utils.get_ocr_roi(
                    point_mean, R_tcp[:, 1], R_tcp[:, 0], self.WIDTH, self.LENGTH
                )

                self.publish_plate_center(
                    point_center_edge_1,
                    point_center_edge_2,
                    point_center_edge_1_inner,
                    point_center_edge_2_inner,
                )

                p_img_edge_1 = utils.transform_base_img(
                    point_center_edge_1, ee_pose, image_width, image_height
                )
                p_img_edge_2 = utils.transform_base_img(
                    point_center_edge_2, ee_pose, image_width, image_height
                )
                p_img_inner_1 = utils.transform_base_img(
                    point_center_edge_1_inner, ee_pose, image_width, image_height
                )
                p_img_inner_2 = utils.transform_base_img(
                    point_center_edge_2_inner, ee_pose, image_width, image_height
                )

                utils.cut_out_img(
                    rgb_image_copy,
                    p_img_edge_1[0:2],
                    p_img_edge_2[0:2],
                    p_img_inner_1[0:2],
                    p_img_inner_2[0:2],
                )

                debug_array.append(str(point_base[0]))
                debug_array.append(str(point_base[1]))
                debug_array.append(str(point_base[2]))

                debug_array.append(str(quaternion[0]))
                debug_array.append(str(quaternion[1]))
                debug_array.append(str(quaternion[2]))
                debug_array.append(str(quaternion[3]))

                point_base_copy = copy.deepcopy(point_base)

                radius_list = np.array([0.27, 0.25, 0.22, 0.18, 0.14])
                c = 1

                point_base = copy.deepcopy(point_base_copy)

                self.moxa.setLightOn()

                radius = 0.15
                point_base[2] += 0.15
                points, rotated_quats = self.generate_points(
                    quaternion, point_base, R_tcp[:, 1], normal_z, radius
                )

                self.plot_points_and_tcp(points, rotated_quats, point_mean[0:3])

                self.position_controller.go_to_pose(point_base, quaternion)

                start_counter = time.time()
                for i in range(len(points)):

                    self.TRACK_OCR = True

                    self.position_controller.go_to_pose(points[i], rotated_quats[i])

                    rgb_image_copy_debug = copy.deepcopy(np.asarray(self.rgb_image))

                    depth_data_copy_debug = copy.deepcopy(np.asarray(self.depth_image))

                    segmented_image, segmented_depth = self.plate_detector.segment(
                        rgb_image_copy_debug, depth_data_copy_debug, debug=False
                    )

                    if self.OCR_DONE == True:

                        self.OCR_DONE = False
                        self.ocr_counter = 0
                        self.ocr_detection = []

                        self.position_controller.init_search()
                        return debug_array

                c += 1

                return debug_array
            except Exception as e:
                rospy.logerr(f"Error: {str(e)}")

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()

    def correction_angle_callback(self, msg):
        try:
            self.correction_angle = msg
        except Exception as e:
            rospy.logerr(f"Error processing Correction Angle message: {str(e)}")

    def tf_callback(self, msg):
        try:

            for transform in msg.transforms:

                if (
                    transform.child_frame_id == "marker_24"
                    and transform.header.frame_id == "base"
                ):

                    self.aruco_position = transform.transform

        except Exception as e:
            rospy.logwarn(f"Transformation for AruCo Marker: {str(e)}")

    def ocr_callback(self, msg):
        try:

            if self.TRACK_OCR:
                self.ocr_detection.append(msg.data)
                if self.ocr_counter >= 10:
                    self.OCR_DONE = True
                if msg.data == "False":
                    self.ocr_counter += 1
                else:
                    self.ocr_counter = 0
        except Exception as e:
            rospy.logerr(f"Error processing OCR message: {str(e)}")

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
            self.camera_intrinsics["f"] = msg.K[0]
            self.camera_intrinsics["cx"] = msg.K[2]
            self.camera_intrinsics["cy"] = msg.K[5]
        except Exception as e:
            rospy.logerr(f"Error getting camera info: {str(e)}")


if __name__ == "__main__":
    rospy.init_node("automation_pipeline", anonymous=True)
    pipeline = AutomationPipeline()

    try:
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            pipeline.start()
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")
