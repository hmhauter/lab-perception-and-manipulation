#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import copy
import tf2_ros
import time
from matplotlib import pyplot as plt
from position_controller import PositionControllerUR
import utils
import transforms3d.quaternions as quat
import csv
from std_msgs.msg import Bool
from plate_detector import PlateDetector
from grasp_point_estimator_old import GraspPointEstimator
from geometry_msgs.msg import Point
from std_msgs.msg import String, Float32
from ur_master_thesis.msg import GraspPoints, PlateROI

from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern, RationalQuadratic
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D

class OFRecorder():
    def __init__(self):
        self.experiment = "run8"
        rgb_topic = "/camera/color/image_raw"
        img_type = Image

        self.rgb_image = None

        self.bridge = CvBridge()
        rospy.Subscriber(rgb_topic, img_type, self.rgb_image_callback)

        self.frame1 = None
        self.frame2 = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.plate_detector = PlateDetector("/home/apo/Documents/MasterThesis/adaptive-lab-automation/datasets/runs/segment/train/weights/best.pt")
        
        self.tx = 0
        self.ty = 0
        self.tz = 0

        self.qx = 0
        self.qy = 0
        self.qz = 0
        self.qw = 0

        self.position_controller = PositionControllerUR()
        self.start_recording = False

        # Polynommial Model 
        degree = 2
        self.polynomial_model = Pipeline([
            ('polynomialfeatures', PolynomialFeatures(degree=3)),  # Change degree as needed
            ('linearregression', LinearRegression())
        ])
        # Gaussian Process Model
        kernel = WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)) + RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e5)) 
        # kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e7)) + WhiteKernel()
        # kernel = 1.0 *  WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)) # RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e5)) #

        self.gaussian_model = GaussianProcessRegressor(kernel=kernel, random_state=0)
        self.X = np.array([])
        self.Y = np.array([])

        self.scaler = StandardScaler()

        self.roi = None
        self.plate_center_available = False
        rospy.Subscriber("/plate/center", PlateROI, self.point_callback)
        rospy.Subscriber('/record/status', Bool, self.record_callback)

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
        # Calculate the mean brightness
        average_brightness = np.mean(gray_image)
        return average_brightness
    
    def segment_img(self, rgb_img):
        height, width = rgb_img.shape[:2]
        depth_data_copy = np.ones((height, width), dtype=np.uint8)
        segmented_image, segmented_depth, bbox = self.plate_detector.segment_and_detect(rgb_img, depth_data_copy, debug=False)
        return segmented_image, bbox

    def calculate_highlights(self, img_masked, img):

        histogram, bin_edges = np.histogram(img_masked, bins=256, range=(0, 256))

        smoothed_histogram = gaussian_filter1d(histogram, sigma=2)
        peaks, _ = find_peaks(smoothed_histogram, distance=10)
        print("Peaks: ")
        print(peaks)
        if len(peaks) > 0:
            threshold = peaks[-1]
        else:
            rospy.logerr("ERROR Histogram")
            threshold = 0  # Default to 0 if no peaks are found
        
        max_peak = np.argmax(smoothed_histogram)
        threshold_max = bin_edges[max_peak]
        print("Threshold MAX")
        print(threshold_max)
        
        # Find the top 5 peak values in the histogram
        num_peaks = 50
        peaks = np.argpartition(histogram, -num_peaks)[-num_peaks:]
        peaks = np.sort(peaks)  # Sort the peak indices in ascending order

        # Create a mask for pixel values above the threshold
        above_threshold_mask = img > threshold

        print("THRESHOLD")
        print(threshold)

        # Highlight the values above the threshold in the original image
        highlighted_image = np.ones_like(img) * 255
        highlighted_image[~above_threshold_mask] = 0  # Set all other pixels to 0
        colors = {0: (231.999/255, 63.010/255, 72.012/255), 1: (0.000, 135.992/255, 52.989/255), 2: (251.991/255, 117.989/255, 51.995/255), 3: (120.997/255, 35.011/255, 142.009/255)}


        #################### DEBUGGING ############## 
        # plt.figure(1)
        # plt.clf()  # Clear the previous plot
        # # plt.plot(bin_edges[peaks], smoothed_histogram[peaks], "x", color="b", label='Peaks')
        # plt.plot(smoothed_histogram, color=colors[0])
        # plt.xlabel("Pixel Value")
        # plt.ylabel("Frequency")
        # plt.title("Histogram")

        # # Save the plot as an image and display it with OpenCV
        # plt.savefig(f'/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/run5/histogram_{self.histogram_it}.png')
        # plt.figure(2)
        # plt.clf()  # Clear the previous plot
        # plt.plot(bin_edges[peaks], smoothed_histogram[peaks], "x", color=colors[0], label='Peaks')
        # # plt.plot(smoothed_histogram,1 color="k")
        # plt.xlabel("Pixel Value")
        # plt.ylabel("Frequency")
        # plt.title("Histogram Smooth")

        # # Save the plot as an image and display it with OpenCV
        # plt.savefig(f'/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/run5/histogramSmooth_{self.histogram_it}.png')
        ####################################
        self.histogram_it += 1
        return highlighted_image
    
    # def cv2_to_tensor(self, image):
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    #     image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    #     image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    #     return image_tensor

    def write_to_csv(self, data):
        filename = f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/{self.experiment}/record.csv"
        # filename = "/home/apo/catkin_ws/src/ur_master_thesis/src/ObjectiveFunction/main/main.csv"
        # Open the CSV file in append mode
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the data to the CSV file
            writer.writerow(data)

    def create_roi_mask(self, image, roi_corners):
        image_gray = image # cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(image_gray.shape, dtype=np.uint8)

        roi_corners = np.array(roi_corners, dtype=np.int32)


        mask = cv2.fillPoly(mask, [roi_corners], 255)
   
        image_masked = cv2.bitwise_and(image_gray, image_gray, mask=mask)
        return image_masked

    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    def calculate_distance_from_border(self, bbox, image_width, image_height):
        x, y, width, height = bbox[0].cpu().numpy()
        distance_left = x
        distance_right = image_width - (x + width)
        distance_top = y
        distance_bottom = image_height - (y + height)
        margin = 10
        x_margin_condition = x >= margin and x + width <= image_width - margin
        print("Is x on image?")
        print(x_margin_condition)
        y_margin_condition = y >= margin and y + height <= image_height - margin
        print("Is y on image?")
        print(y)
        print(y+height)
        print(image_height - margin)
        print(y_margin_condition)
        if x_margin_condition:
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
        # alpha = 0.5
        # beta = 0.3
        # gamma = 0.8
        # feta = 60

        alpha = 0.5
        beta = 0.0
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
        # plt.figure(1)
        # plt.imshow(gray, cmap="gray")
        # plt.imsave(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/run5/gray_img__record{self.obj_fun_iteration}.jpg", gray)
        # plt.figure(2)
        # plt.imshow(highlighted_image, cmap="gray")
        # plt.imsave(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/run5/highlited_img_{self.obj_fun_iteration}.jpg", highlighted_image)
        # plt.figure(3)
        # plt.imshow(roi_mask_highlighted, cmap="gray")
        # plt.imsave(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/run5/roi_mask_highlighted_{self.obj_fun_iteration}.jpg", roi_mask_highlighted)
        # plt.figure(4)
        # plt.imshow(roi_mask, cmap="gray")
        # plt.imsave(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/run5/roi_mask_{self.obj_fun_iteration}.jpg", roi_mask)
        # plt.figure(5)
        # plt.imshow(combined_img, cmap="gray")
        # plt.imsave(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/run5/combined_img_{self.obj_fun_iteration}.jpg", combined_img)
        self.obj_fun_iteration += 1
        # print("Objective function returns: ", (obj_fun_img))
        return obj_fun_img, combined_img, gray

    def get_ee_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform("base", "tool0_controller", rospy.Time(0), rospy.Duration(1.0))
            return transform.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to get transform: %s", e)

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
    
    def fit_polynomial_model(self):

        if self.start_recording == True and self.plate_center_available == True: 
        # if self.plate_center_available == True: 
            ee_pose = self.get_ee_pose()
            print("EE Pose")
            print(ee_pose)
            # Update X and Y 
            rgb_img = copy.deepcopy(self.rgb_image)
            roi = self.get_roi(ee_pose)

            if roi is not None:
                segmented_image, bbox = self.segment_img(rgb_img)
                output_value, combined_img, gray = self.calc_objective_function(segmented_image, roi, bbox)
                print("Objective function value")
                print(output_value)
                # Create a rotation object from the quaternion
                r = Rotation.from_quat([ee_pose.rotation.x,
                        ee_pose.rotation.y,
                        ee_pose.rotation.z,
                        ee_pose.rotation.w])
                
                # Convert the rotation to Euler angles
                roll, pitch, yaw = r.as_euler('xyz', degrees=False)
                # New data point
                new_X = np.array([ee_pose.translation.x, ee_pose.translation.y, ee_pose.translation.z, roll, pitch, yaw])  
                new_y = output_value
                print(new_X)
                print(new_y)

                self.X = np.vstack([self.X, new_X]) if self.X.size else np.array([new_X])
                self.Y = np.append(self.Y, new_y)

                cv2.imwrite(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/{self.experiment}/obj_img_{self.iteration}.jpg", combined_img)
                cv2.imwrite(f"/home/apo/catkin_ws/src/ur_master_thesis/src/log_gd/{self.experiment}/gray_img_{self.iteration}.jpg", gray)

                csv_array = np.array([self.iteration, 
                                    ee_pose.translation.x, 
                                    ee_pose.translation.y, 
                                    ee_pose.translation.z,
                                    ee_pose.rotation.x,
                                    ee_pose.rotation.y,
                                    ee_pose.rotation.z,
                                    ee_pose.rotation.w, 
                                    output_value])
                
                self.write_to_csv(csv_array)

                self.iteration = self.iteration + 1

            else:
                print("No ROI")
    
    def plot_model_poly(self):
        print("Start plotting....")

        x_range = np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), 50)
        y_range = np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)

        # Calculate mean values for z, roll, pitch, and yaw
        z_mean = np.mean(self.X[:, 2])
        roll_mean = np.mean(self.X[:, 3])
        pitch_mean = np.mean(self.X[:, 4])
        yaw_mean = np.mean(self.X[:, 5])

        # Create arrays for mean values
        z_grid = np.full_like(X_grid, z_mean)
        roll_grid = np.full_like(X_grid, roll_mean)
        pitch_grid = np.full_like(X_grid, pitch_mean)
        yaw_grid = np.full_like(X_grid, yaw_mean)

        # Concatenate X_grid, Y_grid, z_grid, roll_grid, pitch_grid, and yaw_grid
        XYZRPY_grid_transformed = np.stack([X_grid, Y_grid, z_grid, roll_grid, pitch_grid, yaw_grid], axis=-1)

        # Flatten and reshape for prediction
        XYZRPY_grid_transformed_flat = XYZRPY_grid_transformed.reshape(-1, 6)

        # Predict objective function values
        objective_function_values = self.polynomial_model.predict(XYZRPY_grid_transformed_flat).reshape(X_grid.shape)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(X_grid, Y_grid, objective_function_values, color='b', alpha=0.5, rstride=100, cstride=100)

        # Plot the original data points
        ax.scatter(self.X[:, 0], self.X[:, 1], self.Y, color='r')

        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Objective Function')


        plt.savefig(f'/home/apo/catkin_ws/src/ur_master_thesis/src/polynomial_model/model_{self.iteration}.png')
        self.iteration = self.iteration + 1


    def fit_gaussian_model(self):
        if self.plate_center_available == True: 
            print("Start fit gaussian model")
            ee_pose = self.get_ee_pose()
            print("EE Pose")
            print(ee_pose)
            # Update X and Y 
            rgb_img = copy.deepcopy(self.rgb_image)
            roi = self.get_roi(ee_pose)
            print("ROI")
            print(roi)
            segmented_image, bbox = self.segment_img(rgb_img)
            output_value = self.calc_objective_function(segmented_image, roi)
            print("Objective function value")
            print(output_value)
            # Create a rotation object from the quaternion
            r = Rotation.from_quat([ee_pose.rotation.x,
                    ee_pose.rotation.y,
                    ee_pose.rotation.z,
                    ee_pose.rotation.w])
            
            # Convert the rotation to Euler angles
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            # New data point
            new_X = np.array([ee_pose.translation.x, ee_pose.translation.y, ee_pose.translation.z, roll, pitch, yaw])  
            new_y = output_value
            print(new_X)
            print(new_y)

            self.X = np.vstack([self.X, new_X]) if self.X.size else np.array([new_X])
            self.Y = np.append(self.Y, new_y)

            X_transformed = self.scaler.fit_transform(self.X)

            print(self.X)
            print(X_transformed)
            print(self.Y)

            # Update the model with the new data point
            self.gaussian_model = self.gaussian_model.fit(X_transformed, self.Y)
            rospy.loginfo("------------ Fit Gaussian Model ------------")

            print(self.gaussian_model)

        

            self.plot_model()

    def plot_model(self):
        print("Start plt gaussian model")
        # Fix dimensions z, roll, pitch, yaw at their mean values
        fixed_dims = np.mean(self.X[:, 2:], axis=0)
        
        # Create a meshgrid for the X-Y plane
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        x_values, y_values = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        # Prepare the input for prediction by combining the meshgrid with fixed dimensions
        X_meshgrid = np.c_[x_values.ravel(), y_values.ravel()]
        fixed_dims_repeated = np.tile(fixed_dims, (X_meshgrid.shape[0], 1))
        X_full = np.hstack((X_meshgrid, fixed_dims_repeated))
        
        # Make predictions on the generated meshgrid
        y_pred, sigma = self.gaussian_model.predict(X_full, return_std=True)
        
        # Visualize the model in the X-Y plane
        plt.figure(figsize=(10, 8))
        
        # Plot the contour map for predictions
        contour = plt.contourf(x_values, y_values, y_pred.reshape(x_values.shape), alpha=0.8, cmap='viridis')
        plt.colorbar(contour, label='Predicted Data')
        
        # Plot the sampled data points
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, edgecolor='k', cmap='coolwarm', label='Sampled Data')
        plt.colorbar(scatter, label='Sampled Data')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Gaussian Process Model in X-Y Plane')
        plt.legend()

        plt.savefig(f'/home/apo/catkin_ws/src/ur_master_thesis/src/gaussian_model/model_{self.iteration}.png')

        self.iteration = self.iteration + 1





    # # Function to apply translation to the image
    # def apply_translation(self, image, tx, ty):
    #     # Define the camera motion parameters (e.g., translation in x and y)
    #     tx = torch.tensor([0.0], requires_grad=True)
    #     ty = torch.tensor([0.0], requires_grad=True)
    #     theta = torch.tensor([[[1, 0, tx.item()], [0, 1, ty.item()]]], dtype=torch.float)  # Affine transformation matrix
    #     grid = F.affine_grid(theta, image.size(), align_corners=False)
    #     translated_image = F.grid_sample(image, grid, align_corners=False)
    #     return translated_image


    def get_camera_pose(self):
        try:
            # Wait for the transform to become available
            transform = self.tf_buffer.lookup_transform("base", "camera_color_optical_frame", rospy.Time(0), rospy.Duration(1.0))
            return transform.transform
        except:
            return None

    

    def record_callback(self, msg):
        try:
            print("Got record indicator")
            self.start_recording = msg.data
            if self.start_recording == False:
                self.plate_center_available = False
        except Exception as e:
            rospy.logerr("Error record callback: %s", str(e))


    def point_callback(self, msg):
        print("Point Callback")
        try:
            print("Got plate center")
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
    

if __name__ == '__main__':
    print("Start Light Estimator")
    rospy.init_node('obj_fun_recorder', anonymous=True)
    obj_fun_recorder = OFRecorder()

    rospy.sleep(0.2)

    # # Spin to keep the script alive
    try:
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            obj_fun_recorder.fit_polynomial_model()
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")