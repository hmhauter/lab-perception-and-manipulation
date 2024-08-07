#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np
import utils
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage

"""""
This node subscribes to the /tf topic to receive transformation messages between frames "base" and "tool0_controller", 
processes these transformations into rotation matrices and translations, and then computes and broadcasts transformations 
of a camera frame ("camera_color_optical_frame") relative to both "base" and "base_link" frames.
"""

transformation_matrix = np.identity(4)


def base_tool0_callback(msg):
    'Callback function for processing transformation messages between "base" and "tool0_controller" frames from /tf.'
    for transform in msg.transforms:
        if (
            transform.header.frame_id == "base"
            and transform.child_frame_id == "tool0_controller"
        ):

            translation = transform.transform.translation
            rotation = transform.transform.rotation

            translation = [translation.x, translation.y, translation.z]
            rotation = [rotation.x, rotation.y, rotation.z, rotation.w]

            global transformation_matrix, transformation_matrix_link

            rotation_matrix = Rotation.from_quat(rotation).as_matrix()

            transformation_matrix = np.identity(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            transformation_matrix_link = np.identity(4)
            transformation_matrix_link[:3, :3] = rotation_matrix
            transformation_matrix_link[:3, 3] = translation
            R_z_180 = np.array(
                [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
            transformation_matrix_link = R_z_180 @ transformation_matrix_link

            publish_camera_to_base_transform(tf_broadcaster)


def publish_camera_to_base_transform(tf_broadcaster):
    'Publishes transformations of the camera frame relative to "base" and "base_link".'

    global transformation_matrix, transformation_matrix_link

    camera_tranform = utils.translate_to_matrix()

    transformation = transformation_matrix @ camera_tranform
    t = TransformStamped()
    q = Rotation.from_matrix(transformation[:3, :3]).as_quat()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "base"
    t.child_frame_id = "camera_color_optical_frame"
    t.transform.translation.x = transformation[0, 3]
    t.transform.translation.y = transformation[1, 3]
    t.transform.translation.z = transformation[2, 3]

    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    tf_broadcaster.sendTransform(t)

    transformation_link = transformation_matrix_link @ camera_tranform
    tl = TransformStamped()
    ql = Rotation.from_matrix(transformation_link[:3, :3]).as_quat()
    tl.header.stamp = rospy.Time.now()
    tl.header.frame_id = "base_link"
    tl.child_frame_id = "camera_color_optical_frame"
    tl.transform.translation.x = transformation_link[0, 3]
    tl.transform.translation.y = transformation_link[1, 3]
    tl.transform.translation.z = transformation_link[2, 3]

    tl.transform.rotation.x = ql[0]
    tl.transform.rotation.y = ql[1]
    tl.transform.rotation.z = ql[2]
    tl.transform.rotation.w = ql[3]

    # Publish the transformation
    tf_broadcaster.sendTransform(tl)


if __name__ == "__main__":
    try:
        rospy.init_node("camera_base_tf_publisher")

        rospy.Subscriber("/tf", TFMessage, base_tool0_callback)

        rospy.loginfo(
            "Start /tf broadcaster for camera optical frame to base and camera optical frame to base link..."
        )

        translation_base_tool0 = None
        rotation_base_tool0 = None

        tf_broadcaster = tf2_ros.TransformBroadcaster()

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
