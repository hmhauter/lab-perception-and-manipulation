#!/usr/bin/env python3
import numpy as np

import sys
import rospy
import asyncio
import moveit_commander
import moveit_msgs.msg
import shape_msgs
from geometry_msgs.msg import Pose, Vector3, PoseStamped
from geometry_msgs.msg import WrenchStamped
from lowPassFilter import LowPassFilter
from scipy.spatial.transform import Rotation


class PositionControllerUR:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("manipulator")

        rospy.Subscriber("/wrench", WrenchStamped, self.wrench_callback)
        self.force_ee = 0
        self.filter = LowPassFilter(alpha=0.1)

        self.group.set_goal_orientation_tolerance(0.01)
        self.group.set_goal_tolerance(0.01)
        self.group.set_goal_joint_tolerance(0.01)
        self.group.set_num_planning_attempts(100)
        self.group.set_max_velocity_scaling_factor(0.9)
        self.group.set_max_acceleration_scaling_factor(0.9)

        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=10,
        )

    def wrench_callback(self, msg):
        try:
            self.force_ee = msg.wrench.force
            self.force_ee.z = self.filter.update(self.force_ee.z)
            self.force_ee.y = self.filter.update(self.force_ee.y)
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {str(e)}")

    def get_current_pose(self):
        pose_goal = self.group.get_current_pose().pose

        return pose_goal

    def attach_collision_object(self):
        object_pose = PoseStamped()
        object_pose.pose.position.z += 0.17
        # object_pose.header.frame_id = "tool0"
        # self.scene.add_box("plate", object_pose, size=(0.09, 0.13, 0.02))
        # self.group.attach_object(self.group.get_end_effector_link(), collision_object)
        self.scene.attach_box("tool0", "plate", object_pose, size=(0.09, 0.13, 0.02))
        rospy.loginfo("Collision object attached to the end-effector.")

    def remove_collision_object(self):
        self.scene.remove_world_object("plate")

    def deliver_plate(self, gripper):
        record_force_y = []
        record_force_z = []
        waypoints = []
        pose_goal = self.group.get_current_pose().pose
        rospy.loginfo("Robot pose from position controller RVIZ: %s", str(pose_goal))

        pose_goal.position.z -= 0.05
        waypoints.append(pose_goal)

        rospy.loginfo("Waypoints from position controller RVIZ: %s", str(waypoints))
        import copy

        (plan1, fraction) = self.group.compute_cartesian_path(
            waypoints, 0.01, 0.0, avoid_collisions=False
        )
        traj_scaled = copy.deepcopy(plan1)

        # Scale the time durations by a factor of 2
        for point in traj_scaled.joint_trajectory.points:
            point.time_from_start.secs *= 2
            point.time_from_start.nsecs *= 2
        # Continuously monitor the force while executing the trajectory

        self.group.execute(traj_scaled, wait=False)

        do_run = True
        while do_run == True:
            record_force_y.append(self.force_ee.y)
            record_force_z.append(self.force_ee.z)
            ret1 = False
            if self.force_ee.z < -4 or self.force_ee.y < -4:
                do_run = False

        self.group.stop()
        gripper.move(0, 10, 10)

        # Do not get confused - this is a stupid bu in MoveIT
        # So we move again to stop (instead of caling the stop() function on the group)
        new_pose = self.group.get_current_pose().pose
        new_pose.position.z += 0.02
        (plan2, fraction) = self.group.compute_cartesian_path(
            [new_pose], 0.01, 0.0, avoid_collisions=False
        )
        ret1 = self.group.execute(plan2, wait=True)

        return ret1

    def move_to_plate(self, position):

        waypoints = []
        pose_goal = self.group.get_current_pose().pose
        rospy.loginfo("Robot pose from position controller RVIZ: %s", str(pose_goal))

        pose_goal.position.x = -position[0]
        pose_goal.position.y = -position[1]
        waypoints.append(pose_goal)

        rospy.loginfo("Waypoints from position controller RVIZ: %s", str(waypoints))

        (plan1, fraction) = self.group.compute_cartesian_path(
            waypoints, 0.01, 0.0, avoid_collisions=True
        )
        if fraction == 1.0:
            rospy.loginfo("Cartesian path computed successfully!")

            user_input = input("Do you want to execute the planned path? ")

            if user_input == "o":
                self.group.execute(plan1, wait=True)
                rospy.sleep(4.0)
            else:
                rospy.logwarn("Abort mission...")
        else:
            rospy.logwarn(
                f"Failed to compute the entire Cartesian path. Fraction computed: {fraction}"
            )

    def go_back(self):

        waypoints = []
        pose_goal = self.group.get_current_pose().pose
        rospy.loginfo("Robot pose from position controller RVIZ: %s", str(pose_goal))

        pose_goal.position.z += 0.2
        waypoints.append(pose_goal)

        rospy.loginfo("Waypoints from position controller RVIZ: %s", str(waypoints))

        (plan1, fraction) = self.group.compute_cartesian_path(
            waypoints, 0.01, 0.0, avoid_collisions=True
        )
        if fraction == 1.0:
            rospy.loginfo("Cartesian path computed successfully!")

            self.group.execute(plan1, wait=True)

        else:
            rospy.logwarn(
                f"Failed to compute the entire Cartesian path. Fraction computed: {fraction}"
            )

    def go_to_pose_gd(self, position, orientation):

        waypoints = []
        pose_goal = self.group.get_current_pose().pose

        pose_goal.position.x = -float(position[0])
        pose_goal.position.y = -float(position[1])
        pose_goal.position.z = float(position[2])

        pose_goal.orientation.x = float(orientation[1])
        pose_goal.orientation.y = float(-orientation[0])
        pose_goal.orientation.z = float(-orientation[3])
        pose_goal.orientation.w = float(orientation[2])
        waypoints.append(pose_goal)

        (plan1, fraction) = self.group.compute_cartesian_path(
            waypoints, 0.01, 0.0, avoid_collisions=True
        )
        if fraction == 1.0:

            self.group.execute(plan1, wait=True)

            return True

        else:
            rospy.logwarn(
                f"Failed to compute the entire Cartesian path. Fraction computed: {fraction}"
            )

            return False

    def go_to_pose_intermediate(self, position, orientation):

        waypoints = []
        pose_goal = self.group.get_current_pose().pose

        pose_goal.orientation.x = float(0.655609755263584)
        pose_goal.orientation.y = float(-0.7550993231500154)
        pose_goal.orientation.z = float(0.0007101626735777009)
        pose_goal.orientation.w = float(0.0005972023020047511)
        waypoints.append(pose_goal)

        (plan1, fraction) = self.group.compute_cartesian_path(
            waypoints, 0.01, 0.0, avoid_collisions=True
        )
        if fraction == 1.0:

            self.group.execute(plan1, wait=True)

            return True

        else:
            rospy.logwarn(
                f"Failed to compute the entire Cartesian path. Fraction computed: {fraction}"
            )

            return False

    def go_to_pose(self, position, orientation):

        waypoints = []
        pose_goal = self.group.get_current_pose().pose

        pose_goal.position.x = -float(position[0])
        pose_goal.position.y = -float(position[1])
        pose_goal.position.z = float(position[2])

        pose_goal.orientation.x = float(orientation[0])
        pose_goal.orientation.y = float(orientation[1])
        pose_goal.orientation.z = float(orientation[2])
        pose_goal.orientation.w = float(orientation[3])
        waypoints.append(pose_goal)

        (plan1, fraction) = self.group.compute_cartesian_path(
            waypoints, 0.01, 0.0, avoid_collisions=True
        )
        if fraction == 1.0:

            self.group.execute(plan1, wait=True)

            return True

        else:
            rospy.logwarn(
                f"Failed to compute the entire Cartesian path. Fraction computed: {fraction}"
            )

            return False

    def go_to_light_position(self):

        INIT_POSITION = np.array([-0.12866, -0.3975, 0.20003])

        INIT_ROTATION = np.array([0.99991, -0.013478, 0.0, 0.0])

        R_tcp = Rotation.from_quat(INIT_ROTATION).as_matrix()

        R_tcp = np.array([[1, -0.0012274, 0], [-0.0012274, -1, 0], [0, 0, -1]])

        point_mean = np.array([-0.12825, -0.38226, 0.0090347])

        self.go_to_pose(INIT_POSITION, INIT_ROTATION)
        return point_mean, R_tcp[:, 1], R_tcp[:, 0]

    def deliver_go_to_light_position_rot(self, gripper):

        INIT_POSITION = np.array([-0.12866, -0.3975 + (0.01575 + 0.015), 0.20003])

        INIT_ROTATION = np.array([0.99991, -0.013478, 0.0, 0.0])

        INIT_POSITION[2] += 0.15

        INIT_POSITION[2] -= 0.15
        qx, qy, qz, qw = INIT_ROTATION
        INIT_ROTATION = np.array([-qy, qx, -qw, qz])

        R_tcp = Rotation.from_quat(INIT_ROTATION).as_matrix()

        R_tcp = np.array([[1, -0.0012274, 0], [-0.0012274, -1, 0], [0, 0, -1]])

        point_mean = np.array([-0.12825, -0.38226, 0.0090347])

        self.go_to_pose(INIT_POSITION, INIT_ROTATION)
        gripper.move_and_wait_for_pos(69, 10, 25)
        INIT_POSITION[2] += 0.12
        self.go_to_pose(INIT_POSITION, INIT_ROTATION)
        return point_mean, R_tcp[:, 1], R_tcp[:, 0]

    def deliver_go_to_light_position(self, gripper):

        INIT_POSITION = np.array([-0.12866, -0.3975, 0.20003])

        INIT_ROTATION = np.array([0.99991, -0.013478, 0.0, 0.0])

        R_tcp = Rotation.from_quat(INIT_ROTATION).as_matrix()

        R_tcp = np.array([[1, -0.0012274, 0], [-0.0012274, -1, 0], [0, 0, -1]])

        point_mean = np.array([-0.12825, -0.38226, 0.0090347])

        self.go_to_pose(INIT_POSITION, INIT_ROTATION)
        gripper.move_and_wait_for_pos(69, 10, 25)
        INIT_POSITION[2] += 0.12
        self.go_to_pose(INIT_POSITION, INIT_ROTATION)
        return point_mean, R_tcp[:, 1], R_tcp[:, 0]

    def go_to_detection_position(self):

        INIT_POSITION = np.array(
            [-0.16094683863001932, -0.2673182488317246, 0.4356994386623842]
        )

        INIT_ROTATION = np.array(
            [
                -0.9850548670890956,
                0.0580539184698901,
                0.05320188669078788,
                0.15318684874008034,
            ]
        )
        self.go_to_pose(INIT_POSITION, INIT_ROTATION)

    def init_search(self):
        pose_goal = self.group.get_current_pose().pose

        INIT_POSITION = np.array(
            [-0.22304020587499895, -0.23579385492884822, 0.5772752903096521]
        )
        INIT_ROTATION = np.array(
            [
                -0.9277769605760685,
                0.3531988613996627,
                -0.050195941541194715,
                0.1093656398646637,
            ]
        )
        self.go_to_pose(INIT_POSITION, INIT_ROTATION)

    def go_to_pose_test(self, position):

        waypoints = []
        pose_goal = self.group.get_current_pose().pose

        pose_goal.position.x = -position[0]
        pose_goal.position.y = -position[1]
        pose_goal.position.z = position[2]

        waypoints.append(pose_goal)

        (plan1, fraction) = self.group.compute_cartesian_path(
            waypoints, 0.01, 0.0, avoid_collisions=True
        )
        if fraction == 1.0:

            user_input = input("Do you want to execute the planned path? ")
            print(user_input)
            if user_input == "o":
                self.group.execute(plan1, wait=True)
                rospy.sleep(4.0)
            else:
                print("Abort mission...")
        else:
            print(
                f"Failed to compute the entire Cartesian path. Fraction computed: {fraction}"
            )

        print("============ Waiting while RVIZ displays plan1...")
        rospy.sleep(0.5)

        print("============ Visualizing plan1")
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan1)
        self.display_trajectory_publisher.publish(display_trajectory)
        print("============ Waiting while plan1 is visualized (again)...")
        rospy.sleep(2.0)

    def move_base_joint(self, angle_degrees):
        rospy.loginfo("Move base joint...")

        current_joint_values = self.group.get_current_joint_values()

        current_base_joint_value = current_joint_values[0]
        new_base_joint_value = current_base_joint_value + np.radians(angle_degrees)

        joint_values_to_set = current_joint_values
        joint_values_to_set[0] = new_base_joint_value

        self.group.set_joint_value_target(joint_values_to_set)
        plan = self.group.plan()

        if plan[0]:
            self.group.execute(plan[1])
        else:
            rospy.logwarn("Failed to plan trajectory.")

    def move_wrist1_joint(self, angle_degrees):
        rospy.loginfo("Move wrist1 joint...")

        current_joint_values = self.group.get_current_joint_values()

        current_wrist_joint_value = current_joint_values[3]
        new_wrist_joint_value = current_wrist_joint_value + np.radians(angle_degrees)

        joint_values_to_set = current_joint_values
        joint_values_to_set[3] = new_wrist_joint_value

        self.group.set_joint_value_target(joint_values_to_set)
        plan = self.group.plan()

        if plan[0]:
            self.group.execute(plan[1])
        else:
            rospy.logwarn("Failed to plan trajectory.")

    def move_wrist1_joint(self, angle_degrees):
        rospy.loginfo("Move wrist3 joint...")

        current_joint_values = self.group.get_current_joint_values()

        current_wrist_joint_value = current_joint_values[3]
        new_wrist_joint_value = current_wrist_joint_value + np.radians(angle_degrees)

        joint_values_to_set = current_joint_values
        joint_values_to_set[3] = new_wrist_joint_value

        self.group.set_joint_value_target(joint_values_to_set)
        plan = self.group.plan()

        if plan[0]:
            self.group.execute(plan[1])
        else:
            rospy.logwarn("Failed to plan trajectory.")


if __name__ == "__main__":
    rospy.init_node("poscontroller", anonymous=True)
    pos_controller = PositionControllerUR()
    pos = pos_controller.get_current_pose()

    pos_controller.go_to_detection_position()
