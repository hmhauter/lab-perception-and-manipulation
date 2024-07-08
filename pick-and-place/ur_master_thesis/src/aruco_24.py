#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import cv_bridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import tf2_ros
from scipy.spatial.transform import Rotation

"""""
Node detects ArUco marker on RGB image and streams the pose to /tf
""" ""


class ArucoMarkerDetector:
    def __init__(self):

        rospy.init_node("aruco_marker_detector")

        self.bridge = cv_bridge.CvBridge()
        self.rgb_image = None
        self.camera_info = None
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Parameters
        self.marker_size = 0.05
        self.marker_id = 24

        # Subscribers
        rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        self.rgb_subscriber = rospy.Subscriber(
            rgb_topic, Image, self.rgb_image_callback
        )
        camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera/color/camera_info"
        )
        self.camera_info_subscriber = rospy.Subscriber(
            camera_info_topic, CameraInfo, self.camera_info_callback
        )

        rospy.loginfo("Aruco Marker Detector node initialized.")

    def rgb_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb_image = cv_image
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def aruco_pose_to_homogeneous_transform(self, rvec, tvec):
        "Converts the pose of an ArUco marker into a 4x4 homogeneous transformation matrix."
        rvec_mat, _ = cv2.Rodrigues(rvec)
        homogeneous_transform = np.eye(4)
        homogeneous_transform[:3, :3] = rvec_mat
        homogeneous_transform[:3, 3] = tvec.squeeze()

        return homogeneous_transform

    def transform_aruco_pose_to_base(self, rvec, tvec, camera_pose_base):
        "Transforms the pose of an ArUco marker into the base frame."
        aruco_homogeneous_transform = self.aruco_pose_to_homogeneous_transform(
            rvec, tvec
        )
        rotation_matrix = Rotation.from_quat(
            [
                camera_pose_base.rotation.x,
                camera_pose_base.rotation.y,
                camera_pose_base.rotation.z,
                camera_pose_base.rotation.w,
            ]
        ).as_matrix()

        homogeneous_transform = np.eye(4)
        homogeneous_transform[:3, :3] = rotation_matrix

        homogeneous_transform[:3, 3] = [
            camera_pose_base.translation.x,
            camera_pose_base.translation.y,
            camera_pose_base.translation.z,
        ]

        aruco_pose_base_transformed = np.dot(
            homogeneous_transform, aruco_homogeneous_transform
        )

        aruco_pose_base = TransformStamped()
        aruco_pose_base.header.stamp = rospy.Time.now()
        aruco_pose_base.header.frame_id = "base"
        aruco_pose_base.child_frame_id = "marker_24"
        aruco_pose_base.transform.translation.x = aruco_pose_base_transformed[0, 3]
        aruco_pose_base.transform.translation.y = aruco_pose_base_transformed[1, 3]
        aruco_pose_base.transform.translation.z = aruco_pose_base_transformed[2, 3]

        quaternion = Rotation.from_matrix(aruco_pose_base_transformed[:3, :3]).as_quat()
        aruco_pose_base.transform.rotation.x = quaternion[0]
        aruco_pose_base.transform.rotation.y = quaternion[1]
        aruco_pose_base.transform.rotation.z = quaternion[2]
        aruco_pose_base.transform.rotation.w = quaternion[3]

        return aruco_pose_base

    def my_estimatePoseSingleMarkers(self, corners, marker_size, mtx, distortion):
        "Estimates the pose of a single marker from its detected corners."

        K = np.array(mtx)

        K = K.reshape(3, 3)
        corners = np.array(corners[0], dtype=np.float32)

        marker_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )

        nada, R, t = cv2.solvePnP(
            marker_points, corners, K, distortion, None, None, flags=0
        )

        return R, t, nada

    def get_camera_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "base", "camera_color_optical_frame", rospy.Time(0), rospy.Duration(1.0)
            )
            return transform.transform
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr("Failed to get transform: %s", e)

    def detect_aruco_marker(self):
        "Detects a specific ArUco marker in the RGB image and publishes its pose as a transform to /tf."
        if self.rgb_image is None or self.camera_info is None:
            return

        try:
            camera_pose_base = self.get_camera_pose()
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

            corners, ids, rejected = detector.detectMarkers(self.rgb_image)

            if ids is not None and self.marker_id in ids:
                index = np.where(ids == self.marker_id)[0][0]

                rvec, tvec, _ = self.my_estimatePoseSingleMarkers(
                    corners[index],
                    self.marker_size,
                    self.camera_info.K,
                    np.array(self.camera_info.D),
                )

                aruco_pose_base = self.transform_aruco_pose_to_base(
                    rvec, tvec, camera_pose_base
                )
                # Visualize detection
                # cv2.aruco.drawDetectedMarkers(self.rgb_image, corners)
                # cv2.imshow("ArUco Marker Detection", self.rgb_image)
                # cv2.waitKey(1)

                self.tf_broadcaster.sendTransform(aruco_pose_base)

        except Exception as e:
            rospy.logerr(f"Error detecting ArUco marker: {str(e)}")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.detect_aruco_marker()
            rate.sleep()


if __name__ == "__main__":
    try:
        detector = ArucoMarkerDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
