import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import geometry_msgs.msg

# SO THE PRoBLEM IS THAT OUR ROTAtiON IS DEFINED IN RVIZ FRAME HERE 
# SO PLEaSE ROTATE 180 around Z



def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion

    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

    return rotation_matrix

def choose_minimum_rotation(current_quaternion, desired_quaternion1, desired_quaternion2):
    # Convert quaternions to rotation objects
    current_rot = Rotation.from_quat(current_quaternion)
    desired_rot1 = Rotation.from_quat(desired_quaternion1)
    desired_rot2 = Rotation.from_quat(desired_quaternion2)

    # Calculate relative rotations
    relative_rot1 = desired_rot1 * current_rot.inv()
    relative_rot2 = desired_rot2 * current_rot.inv()

    # Calculate rotation magnitudes
    magnitude1 = relative_rot1.magnitude()
    magnitude2 = relative_rot2.magnitude()

    # Choose the option with the minimum rotation magnitude
    if magnitude1 < magnitude2:
        return desired_quaternion1
    else:
        return desired_quaternion2

def get_tcp_rot(n_z, n_y, current_pose):
    # Extract current orientation from the provided pose
    current_rotation = np.array([current_pose.rotation.x, current_pose.rotation.y, current_pose.rotation.z, current_pose.rotation.w])

    turn_180_z = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    # Calculate Z-axis (Rz)
    # we know that the z axis has to be negativ
    if n_z[2] < 0:
        z_tcp = n_z / np.linalg.norm(n_z)
    else:
        z_tcp = - n_z / np.linalg.norm(n_z)

    z_tcp = np.array([0,0,-1])

    # Calculate Y-axis (Ry)
    n_y[2] = 0
    if n_y[1] < 0:
        y_tcp = n_y / np.linalg.norm(n_y)
    else:
        y_tcp = - n_y / np.linalg.norm(n_y)

    # Calculate X-axis (Rx)
    x_tcp = np.cross(y_tcp, z_tcp)
    x_tcp = x_tcp / np.linalg.norm(x_tcp)

    # Construct the rotation matrices
    R_tcp_1 = np.column_stack((x_tcp, y_tcp, z_tcp))

    # Generate the second option by rotating around the negative of the target y-axis vector
    R_tcp_2 = turn_180_z @ R_tcp_1

    # Convert rotation matrices to rotation objects
    r1 = Rotation.from_matrix(R_tcp_1)
    r2 = Rotation.from_matrix(R_tcp_2)

    # Convert current orientation to rotation object
    current_rot = Rotation.from_quat(current_rotation)

    # Calculate relative rotations
    relative_rot_1 = r1 * current_rot.inv()
    relative_rot_2 = r2 * current_rot.inv()

    # Calculate rotation angles
    angle_1 = relative_rot_1.magnitude()
    angle_2 = relative_rot_2.magnitude()

    # Choose the option with the smallest rotation angle
    # yes this way aroun bc RVIZ has coordinate system turned around 180 around z 
    chosen_rotation = R_tcp_1 if angle_1 > angle_2 else R_tcp_2

    return chosen_rotation, z_tcp, y_tcp

def transform_tool0_gripper():
    transformation = np.eye(4)
    transformation[2, 2] = 0.1776
    return transformation

def get_ocr_roi_2(point_center, y_vector, x_vector, PLATE_WIDTH, PLATE_LENGTH):
    y_vector /= np.linalg.norm(y_vector)
    x_vector /= np.linalg.norm(x_vector)

    point_center_edge = point_center[0:3] - 0.55 * PLATE_LENGTH * y_vector
    point_center_edge_1 = point_center_edge + 0.5 * PLATE_WIDTH * x_vector
    point_center_edge_2 = point_center_edge - 0.5 * PLATE_WIDTH * x_vector
    point_center_edge_inner = point_center[0:3] - 0.35 * PLATE_LENGTH * y_vector
    point_center_edge_1_inner = point_center_edge_inner + 0.5 * PLATE_WIDTH * x_vector
    point_center_edge_2_inner = point_center_edge_inner - 0.5 * PLATE_WIDTH * x_vector

    # p_img_edge_1 = transform_base_img(point_center_edge_1, ee_pose)
    # p_img_edge_2 = transform_base_img(point_center_edge_2, ee_pose)
    # p_img_inner_1 = transform_base_img(point_center_edge_1_inner, ee_pose)
    # p_img_inner_2 = transform_base_img(point_center_edge_2_inner, ee_pose)

    return point_center_edge_1, point_center_edge_2, point_center_edge_1_inner, point_center_edge_2_inner

def get_ocr_roi(point_center, y_vector, x_vector, PLATE_WIDTH, PLATE_LENGTH):
    y_vector /= np.linalg.norm(y_vector)
    x_vector /= np.linalg.norm(x_vector)

    point_center_edge = point_center[0:3] + 0.55 * PLATE_LENGTH * y_vector
    point_center_edge_1 = point_center_edge + 0.5 * PLATE_WIDTH * x_vector
    point_center_edge_2 = point_center_edge - 0.5 * PLATE_WIDTH * x_vector
    point_center_edge_inner = point_center[0:3] + 0.35 * PLATE_LENGTH * y_vector
    point_center_edge_1_inner = point_center_edge_inner + 0.5 * PLATE_WIDTH * x_vector
    point_center_edge_2_inner = point_center_edge_inner - 0.5 * PLATE_WIDTH * x_vector

    # p_img_edge_1 = transform_base_img(point_center_edge_1, ee_pose)
    # p_img_edge_2 = transform_base_img(point_center_edge_2, ee_pose)
    # p_img_inner_1 = transform_base_img(point_center_edge_1_inner, ee_pose)
    # p_img_inner_2 = transform_base_img(point_center_edge_2_inner, ee_pose)

    return point_center_edge_1, point_center_edge_2, point_center_edge_1_inner, point_center_edge_2_inner

def transform_base_img(base_point, ee_pose, image_width, image_height):
    intr = get_camera_intrinsic_matrix()
    extr = translate_to_matrix()
    ee_matr = transformation_to_matrix(ee_pose)
    base_point = np.array(base_point)
    base_point = np.append(base_point, 1.0)


    p_cam = np.linalg.pinv(extr) @ np.linalg.pinv(ee_matr) @ base_point

    p_cam = p_cam[0:3]
    p_img = intr @ p_cam


    p_img /= p_img[2]

    x, y = p_img[0], p_img[1]
    if 0 <= x < image_width and 0 <= y < image_height:
        return p_img
    else:
        return None 

def rotate_image(image, angle, center):
    # Get the image dimensions
    (h, w) = image.shape[:2]
    
    # Rotate the image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated_image


def cut_out_img_rotated(image, p_img_edge_1, p_img_edge_2, p_img_inner_1, p_img_inner_2):
    # Define the four points of the non-straight rectangle
    points = np.array([p_img_edge_1, p_img_edge_2, p_img_inner_1, p_img_inner_2], dtype=np.float32)

    # Calculate the angle of rotation
    # For example, if you want to straighten the box horizontally, you can find the angle between the line connecting top-left and top-right points and the horizontal axis.
    angle = np.arctan2(points[1][1] - points[0][1], points[1][0] - points[0][0]) * 180 / np.pi
  
    # Find the center of the bounding box
    center = ((points[0][0] + points[2][0]) / 2, (points[0][1] + points[2][1]) / 2)

    # Rotate the image
    rotated_image = rotate_image(image, angle, center)

    # Find the dimensions of the bounding box in the rotated image
    xmin = int(min(points[:, 0]))
    xmax = int(max(points[:, 0]))
    ymin = int(min(points[:, 1]))
    ymax = int(max(points[:, 1]))

    mask = np.zeros_like(image)


    # Crop the bounding box region from the rotated image
    cropped_box = rotated_image[ymin:ymax, xmin:xmax]

    return cropped_box



def cut_out_img(image, p_img_edge_1, p_img_edge_2, p_img_inner_1, p_img_inner_2):
    # Define the four points of the non-straight rectangle
    points = np.array([p_img_edge_1, p_img_edge_2, p_img_inner_1, p_img_inner_2], dtype=np.float32)

    # Calculate the angle of rotation
    # For example, if you want to straighten the box horizontally, you can find the angle between the line connecting top-left and top-right points and the horizontal axis.
    angle = np.arctan2(points[1][1] - points[0][1], points[1][0] - points[0][0]) * 180 / np.pi
  
    # Find the center of the bounding box
    center = ((points[0][0] + points[2][0]) / 2, (points[0][1] + points[2][1]) / 2)

    # Rotate the image
    rotated_image = rotate_image(image, angle, center)


    # cv2.imshow('Cut Out Image', rotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find the dimensions of the bounding box in the rotated image
    xmin = int(min(points[:, 0]))
    xmax = int(max(points[:, 0]))
    ymin = int(min(points[:, 1]))
    ymax = int(max(points[:, 1]))

    # Crop the bounding box region from the rotated image
    cropped_box = rotated_image[ymin:ymax, xmin:xmax]

    # cv2.imshow('Cut Out Image', cropped_box)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_box


def transform_pcl_base(pcl_point, ee_pose):
    extr = translate_to_matrix()
    ee_matr = transformation_to_matrix(ee_pose)
    rot = pcl_to_camera_matrix()

    pcl_point = np.array(pcl_point)
    pcl_point = np.append(pcl_point, 1.0)

    point_camera_homogeneous = rot @ pcl_point 

    # from camera to EE frame 
    point_ee_frame = extr @ point_camera_homogeneous


    # present point in EE frame in base frame 
    point_base_frame = ee_matr @ point_ee_frame

    return point_base_frame

def transform_camera_base(camera_point, ee_pose):
    extr = translate_to_matrix()
    ee_matr = transformation_to_matrix(ee_pose)

    pcl_point = np.array(camera_point)
    point_camera_homogeneous = np.append(camera_point, 1.0)

    # from camera to EE frame 
    point_ee_frame = extr @ point_camera_homogeneous

    # NOW IN EE fraMe IS our Chance to GO A BIT IN Y DIrEcTION
    point_ee_frame[1] = point_ee_frame[1] 

    # present point in EE frame in base frame 
    point_base_frame = ee_matr @ point_ee_frame

    return point_base_frame

def straighten_rotation(ee_pose):
    current_rotation = np.array([ee_pose.rotation.x, ee_pose.rotation.y, ee_pose.rotation.z, ee_pose.rotation.w])
    current_orientation = quaternion_to_rotation_matrix(current_rotation)
    # Calculate the rotation angle for the y-axis (angle between the end-effector's y-axis and the base x-axis)
    angle_y = np.arctan2(current_orientation[1, 0], current_orientation[0, 0])

    # Calculate the rotation angle for the x-axis (angle between the end-effector's x-axis and the base y-axis)
    angle_x = np.arctan2(current_orientation[1, 1], current_orientation[1, 0])

    # Create rotation matrices for the x and y axes
    rotation_matrix_x = np.array([[np.cos(angle_x), -np.sin(angle_x), 0],
                                [np.sin(angle_x), np.cos(angle_x), 0],
                                [0, 0, 1]])

    rotation_matrix_y = np.array([[np.cos(angle_y), -np.sin(angle_y), 0],
                                [np.sin(angle_y), np.cos(angle_y), 0],
                                [0, 0, 1]])

    # Combine the rotation matrices for the x and y axes
    # The final rotation matrix will be the product of rotation_matrix_y and rotation_matrix_x
    desired_orientation = np.dot(rotation_matrix_y, rotation_matrix_x)

    rotation_matrix_180_x = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
        ])

    desired_orientation = rotation_matrix_180_x @ desired_orientation
    return desired_orientation

def normalize_angle(angle):
    """
    Normalize the angle to be within the range [-π, π].
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

def angle_camera_base(angle_vec, ee_pose):
  
    angle_vec = np.array(angle_vec)
    angle_vec_homg = np.append(angle_vec, 0)

    ee_matr = transformation_to_matrix(ee_pose)
    extr = translate_to_matrix()
    # from camera to EE frame 
    vec_ee_frame = extr[:3, :3] @ angle_vec_homg


    # present point in EE frame in base frame 
    vec_base_frame = ee_matr[:3, :3] @ vec_ee_frame

    return vec_base_frame

def rotate_pcl_base(pcl_vec, ee_pose):
    extr = translate_to_matrix()
    ee_matr = transformation_to_matrix(ee_pose)
    rot = pcl_to_camera_matrix()

    pcl_vec = np.array(pcl_vec)


    vec_camera_homogeneous = rot[:3, :3] @ pcl_vec 

    # from camera to EE frame 
    vec_ee_frame = extr[:3, :3] @ vec_camera_homogeneous


    # present point in EE frame in base frame 
    vec_base_frame = ee_matr[:3, :3] @ vec_ee_frame
  
    return vec_base_frame

def transform_camera_base(camera_point, ee_pose):
    extr = translate_to_matrix()
    ee_matr = transformation_to_matrix(ee_pose)
    point_camera_homogeneous = np.array(camera_point)
    point_camera_homogeneous = np.append(point_camera_homogeneous, 1.0)

    # from camera to EE frame 
    point_ee_frame = extr @ point_camera_homogeneous


    # present point in EE frame in base frame 
    point_base_frame = ee_matr @ point_ee_frame

    return point_base_frame

def pcl_to_camera_matrix():
    rotation = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
    return rotation

def transform_img_camera(img_point, depth):
    img_point = np.array(img_point)
    img_point = np.append(img_point, 1.0)
    
    camera_matrix = get_camera_intrinsic_matrix()
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    point_camera_homogeneous = depth * (camera_matrix_inv @ img_point)


    return point_camera_homogeneous
  

def transform_img_ee(img_point, depth, ee_pose):
    # depth has to be in m
    extr = translate_to_matrix()
    img_point = np.array(img_point)
    img_point = np.append(img_point, 1.0)
    
    camera_matrix = get_camera_intrinsic_matrix()
  
    ee_matr = transformation_to_matrix(ee_pose)
    ee_matr_inv = np.linalg.inv(ee_matr)

    # from image to camera frame
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    point_camera_homogeneous = depth * (camera_matrix_inv @ img_point)

    point_camera_homogeneous = np.array(point_camera_homogeneous)
    point_camera_homogeneous = np.append(point_camera_homogeneous, 1.0)

    # from camera to EE frame 
    point_ee_frame = extr @ point_camera_homogeneous


    # present point in EE frame in base frame 
    point_base_frame = ee_matr @ point_ee_frame

    return point_base_frame

def get_camera_intrinsic_matrix():
    # camera_matrix = np.array([[607.969482421875, 0.0, 319.28424072265625], [0.0, 607.1234130859375, 237.84567260742188], [0,0,1]])
    camera_matrix = np.array([[911.9542236328125, 0.0, 638.9263305664062], [0.0, 910.6851196289062, 356.7685241699219], [0.0, 0.0, 1.0]]
)
    return camera_matrix 

def transformation_to_matrix(transform):
    translation = np.array([transform.translation.x,
                            transform.translation.y,
                            transform.translation.z])
    
    rotation = np.array([transform.rotation.x,
                         transform.rotation.y,
                         transform.rotation.z,
                         transform.rotation.w])
    
    rotation_matrix = Rotation.from_quat(rotation).as_matrix()

    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix

def translate_to_matrix():

    # ROTATION AND TRANSLATION with light ring 
    translation = np.array([-0.03652509505397746, -0.06080515980821437, 0.14232743585028257])
    rotation = np.array([0.011849472815413993, -0.009825036669174891, -0.012442946593201016, 0.9998040966752062])


    # Convert quaternion to rotation matrix
    rotation_matrix = Rotation.from_quat(rotation).as_matrix()

    # Create transformation matrix
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation


    return transformation_matrix

def get_tcp_rot_side(n_z, n_y, current_pose):
    # Extract current orientation from the provided pose
    current_rotation = np.array([current_pose.rotation.x, current_pose.rotation.y, current_pose.rotation.z, current_pose.rotation.w])

    turn_180_z = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    
    n_y[2] = 0
    if n_y[2] > 0:
        z_tcp = n_y / np.linalg.norm(n_y)
    else:
        z_tcp = - n_y / np.linalg.norm(n_y)



    y_tcp = np.array([0,0,-1])

    # Calculate X-axis (Rx)
    x_tcp = np.cross(y_tcp, z_tcp)
    x_tcp = x_tcp / np.linalg.norm(x_tcp)

    # Construct the rotation matrices
    R_tcp_1 = np.column_stack((x_tcp, y_tcp, z_tcp))

    # Generate the second option by rotating around the negative of the target y-axis vector
    R_tcp_2 = turn_180_z @ R_tcp_1

    # Convert rotation matrices to rotation objects
    r1 = Rotation.from_matrix(R_tcp_1)
    r2 = Rotation.from_matrix(R_tcp_2)

    # Convert current orientation to rotation object
    current_rot = Rotation.from_quat(current_rotation)

    # Calculate relative rotations
    relative_rot_1 = r1 * current_rot.inv()
    relative_rot_2 = r2 * current_rot.inv()

    # Calculate rotation angles
    angle_1 = relative_rot_1.magnitude()
    angle_2 = relative_rot_2.magnitude()

    # Choose the option with the smallest rotation angle
    # yes this way aroun bc RVIZ has coordinate system turned around 180 around z 
    chosen_rotation = R_tcp_1 if angle_1 > angle_2 else R_tcp_2

    return chosen_rotation, z_tcp, y_tcp
def get_tcp_rot_above(n_z, n_y, current_pose):
    # Extract current orientation from the provided pose
    current_rotation = np.array([current_pose.rotation.x, current_pose.rotation.y, current_pose.rotation.z, current_pose.rotation.w])

    turn_180_z = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    # Calculate Z-axis (Rz)
    # we know that the z axis has to be negativ
    if n_z[2] < 0:
        z_tcp = n_z / np.linalg.norm(n_z)
    else:
        z_tcp = - n_z / np.linalg.norm(n_z)
 
    z_tcp = np.array([0,0,-1])

    # Calculate Y-axis (Ry)
    n_y[2] = 0
    if n_y[1] > 0:
        y_tcp = n_y / np.linalg.norm(n_y)
    else:
        y_tcp = - n_y / np.linalg.norm(n_y)

    # Calculate X-axis (Rx)
    x_tcp = np.cross(y_tcp, z_tcp)
    x_tcp = x_tcp / np.linalg.norm(x_tcp)

    # Construct the rotation matrices
    R_tcp_1 = np.column_stack((x_tcp, y_tcp, z_tcp))

    # Generate the second option by rotating around the negative of the target y-axis vector
    R_tcp_2 = turn_180_z @ R_tcp_1

    # Convert rotation matrices to rotation objects
    r1 = Rotation.from_matrix(R_tcp_1)
    r2 = Rotation.from_matrix(R_tcp_2)

    # Convert current orientation to rotation object
    current_rot = Rotation.from_quat(current_rotation)

    # Calculate relative rotations
    relative_rot_1 = r1 * current_rot.inv()
    relative_rot_2 = r2 * current_rot.inv()

    # Calculate rotation angles
    angle_1 = relative_rot_1.magnitude()
    angle_2 = relative_rot_2.magnitude()

    # Choose the option with the smallest rotation angle
    # yes this way aroun bc RVIZ has coordinate system turned around 180 around z 
    chosen_rotation = R_tcp_1 if angle_1 > angle_2 else R_tcp_2




    return chosen_rotation, z_tcp, y_tcp

def get_ee_gripper_transformation_above():
    # Translation vector
    translation = np.array([0.0, 0.01525, 0.17])

    # Create transformation matrix
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, 3] = translation

    return transformation_matrix 

def get_ee_gripper_transformation_side():
    # Translation vector
    translation = np.array([0.0, 0.05, 0.1413])

    # Create transformation matrix
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, 3] = translation

    return transformation_matrix 


def analyze_bounding_box(bbox):
    margin = 70
    bbox_center_x, bbox_center_y, bbox_width, bbox_height = bbox
    image_width = 1280
    image_height = 720
    # Calculate bounding box coordinates
    x_min = bbox_center_x - bbox_width / 2
    y_min = bbox_center_y - bbox_height / 2
    x_max = bbox_center_x + bbox_width / 2
    y_max = bbox_center_y + bbox_height / 2

    # Check if the bounding box is sufficiently away from the edges
    if (x_min > margin and y_min > margin and
            x_max < image_width - margin and y_max < image_height - margin):
        inside = True
    else:
        inside = False

    # Check if the bounding box is more on the right or left side
    if bbox_center_x < image_width / 2:
        side = "left"
    else:
        side = "right"

    # Check if the bounding box is too much up or down
    if bbox_center_y < image_height / 2:
        vertical_direction = "up"
    else:
        vertical_direction = "down"

    # Check if the bounding box is too much in x or y direction
    if x_min < margin or x_max > image_width - margin:
        x_direction = "too much"
    else:
        x_direction = "appropriate"

    if y_min < margin or y_max > image_height - margin:
        y_direction = "too much"
    else:
        y_direction = "appropriate"

    return inside, side, x_direction, y_direction, vertical_direction

