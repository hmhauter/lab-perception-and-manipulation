#!/usr/bin/env python3

import rospy
import numpy as np
import copy
import cv2
import torch
import torch.optim as optim
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from plate_detector import PlateDetector
from position_controller import PositionControllerUR
from shapely import geometry
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
import std_msgs.msg
from ur_master_thesis.msg import PlateROI
from cv_bridge import CvBridge


class GradientDescent():
    def __init__(self):
        self.plate_detector = PlateDetector("/home/apo/Documents/MasterThesis/adaptive-lab-automation/datasets/runs/segment/train/weights/best.pt")
        self.obj_frame = None
        self.position_controller = PositionControllerUR()

        rospy.Subscriber("camera/color/image_raw", Image, self.image_callback)
        rospy.Subscriber("/plate/center", PlateROI, self.point_callback)

        self.roi = None

    def calc_objective_function(self, segmented_img, roi):
        # TODO: you prbl have to normalize the process here 
        # how can we handle constraints? -> use penalty to make sure function stays differentiable 

        alpha = 0.1
        beta = 0.6
        gamma = 0.8

        # Convert the image to grayscale
        gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        # Apply the mask to extract the relevant parts of the grayscale image
        image_cut = gray[mask]

        highlighted_image = self.calculate_highlights(image_cut, gray)

        obj_fun_sum = alpha * gray + beta * highlighted_image 

        # now weight the pixels that are in the ROI more 
        roi_mask = self.create_roi_mask(segmented_img, roi)

        # cut out ROI from image -> multiply ROI with more weight and then add it to overall image 
        obj_fun_img = obj_fun_sum + gamma * roi_mask
        # plt.figure(1)
        # plt.imshow(gray, cmap="gray")
        # plt.figure(2)
        # plt.imshow(highlighted_image, cmap="gray")
        # plt.figure(3)
        # plt.imshow(obj_fun_sum, cmap="gray")
        # plt.figure(4)
        # plt.imshow(roi_mask, cmap="gray")
        # plt.figure(5)
        # plt.imshow(obj_fun_img, cmap="gray")
        # plt.show()

        return obj_fun_img


    def create_roi_mask(self, image, roi_corners):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(image_gray.shape, dtype=np.uint8)

        roi_corners = np.array(roi_corners, dtype=np.int32)


        mask = cv2.fillPoly(mask, [roi_corners], 255)
   
        image_masked = cv2.bitwise_and(image_gray, image_gray, mask=mask)
        return image_masked

    def apply_mask(self, image, mask):
        # Use the mask to set pixels outside the ROI to zero
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image

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


        #################### DEBUGGING ############## 
        # plt.figure(1)
        # plt.clf()  # Clear the previous plot
        # # plt.plot(bin_edges[peaks], smoothed_histogram[peaks], "x", color="b", label='Peaks')
        # plt.plot(smoothed_histogram, color="k")
        # plt.xlabel("Pixel Value")
        # plt.ylabel("Frequency")
        # plt.title("Histogram")

        # # Save the plot as an image and display it with OpenCV
        # plt.savefig('/tmp/histogram.png')
        # histogram_image = cv2.imread('/tmp/histogram.png')
        # cv2.imshow('Histogram', histogram_image)
        # cv2.waitKey(1)
        ####################################

        return highlighted_image

    def cv2_to_tensor(self, image):
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        tensor = torch.tensor(image, requires_grad=True)
        image_tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        return image_tensor
    

    def DEPRECATED_compute_gradient(self, image_difference, tx, ty, tz, qx, qy, qz, qw):
        # Compute gradients using automatic differentiation
        gradient_tx = torch.autograd.grad(image_difference.sum(), tx, create_graph=True)[0]
        gradient_ty = torch.autograd.grad(image_difference.sum(), ty, create_graph=True)[0]
        gradient_tz = torch.autograd.grad(image_difference.sum(), tz, create_graph=True)[0]
        gradient_qx = torch.autograd.grad(image_difference.sum(), qx, create_graph=True)[0]
        gradient_qy = torch.autograd.grad(image_difference.sum(), qy, create_graph=True)[0]
        gradient_qz = torch.autograd.grad(image_difference.sum(), qz, create_graph=True)[0]
        gradient_qw = torch.autograd.grad(image_difference.sum(), qw, create_graph=True)[0]
        return gradient_tx, gradient_ty, gradient_tz, gradient_qx, gradient_qy, gradient_qz, gradient_qw

    def segment_img(self, rgb_img):
        height, width = rgb_img.shape[:2]
        depth_data_copy = np.ones((height, width), dtype=np.uint8)
        segmented_image, segmented_depth = self.plate_detector.segment(rgb_img, depth_data_copy, debug=False)
        return segmented_image

    
    def store_first_image(self, rgb_img, roi):
        segmented_img = self.segment_img(rgb_img)

        # Convert the image to grayscale
        gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        # Apply the mask to extract the relevant parts of the grayscale image
        image_cut = gray[mask]

        obj_func_img = self.calc_objective_function(segmented_img, roi)
        image_tensor = self.cv2_to_tensor(obj_func_img)
        self.obj_frame = image_tensor


    def gradient_descent(self, camera_pose):  
        LEARNING_RATE = 0.01 
        # now again with the Adam optimizer instead
        translation = torch.tensor([
            camera_pose.translation.x,
            camera_pose.translation.y,
            camera_pose.translation.z
        ], requires_grad=True)
        quaternion = torch.tensor([
            camera_pose.rotation.x,
            camera_pose.rotation.y,
            camera_pose.rotation.z,
            camera_pose.rotation.w
        ], requires_grad=True)

        # Optimizer to adjust the translation and quaternion rotation parameters
        optimizer = optim.Adam([translation, quaternion], lr=LEARNING_RATE)

        # Convergence parameters
        tolerance = 1e-4  # Tolerance for convergence
        max_iterations = 1000  # Maximum number of iterations to prevent infinite loop
        previous_objective_value = float('-inf')

        # Optimization loop with convergence check
        for iteration in range(max_iterations):
            optimizer.zero_grad()  # Clear previous gradients

            # Normalize the quaternion to maintain valid rotation
            norm_quaternion = quaternion / quaternion.norm()

            # Move the camera based on current translation and quaternion parameters
            self.position_controller.go_to_pose(translation.data.numpy(), norm_quaternion.data.numpy())
            
            # Capture the current image after moving the camera
            image = copy.deepcopy(self.rgb)
            
            # Compute the objective function value
            objective_value = objective_function(image)
            
            # Since we want to maximize the objective, we minimize the negative objective
            loss = -objective_value
            loss.backward()  # Compute gradients
            optimizer.step()  # Update the parameters

            # Check for convergence
            if abs(objective_value.item() - previous_objective_value) < tolerance:
                print(f"Converged after {iteration} iterations")
                break
            
            previous_objective_value = objective_value.item()
            
            # Print the progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Objective Value: {objective_value.item()}, Translation: {translation.data}, Quaternion: {norm_quaternion.data}")

        else:
            print("Reached maximum iterations without convergence")

        print("Optimization completed")
        print(f"Final Translation: {translation.data}, Final Quaternion: {norm_quaternion.data}")


    # def deprecated_gradient_descent(self, camera_pose, rgb_img, roi):   
    # THIS USES THE AUTODIFF FUNCTION (WE DO NOT WANT THAT) 
    #     is_converged = False 
    #     CONVERGENCE_CRITERIUM = 0.001
    #     tx = torch.tensor([camera_pose.translation.x], requires_grad=True)
    #     ty = torch.tensor([camera_pose.translation.y], requires_grad=True)
    #     tz = torch.tensor([camera_pose.translation.z], requires_grad=True)

    #     qx = torch.tensor([camera_pose.rotation.x], requires_grad=True)
    #     qy = torch.tensor([camera_pose.rotation.y], requires_grad=True)
    #     qz = torch.tensor([camera_pose.rotation.z], requires_grad=True)
    #     qw = torch.tensor([camera_pose.rotation.w], requires_grad=True)

    #     segmented_img = self.segment_img(rgb_img)
    #     print("Segmented")
    #     print(segmented_img)
        

    #     # Convert the image to grayscale
    #     gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    #     image1_tensor = self.obj_frame
    #     obj_func_img = self.calc_objective_function(segmented_img, roi)
    #     image2_tensor = self.cv2_to_tensor(obj_func_img)

    #     img_difference = image1_tensor - image2_tensor
    #     print(img_difference.sum())
    #     print("Image differenvce")
    #     print(img_difference)
    #     plt.figure(1)
    #     plt.imshow(image1_tensor.detach().squeeze().numpy(), cmap="gray")
    #     plt.figure(2)
    #     plt.imshow(image2_tensor.detach().squeeze().numpy(), cmap="gray")

    #     plt.figure(3)
    #     plt.imshow(((image1_tensor - image2_tensor).detach().squeeze().numpy() * 255).astype(np.uint8), cmap="gray")
    #     plt.show()

    #     # Compute gradient of image difference with respect to camera motion parameters
    #     gradient_tx, gradient_ty, gradient_tz, gradient_qx, gradient_qy, gradient_qz, gradient_qw = self.compute_gradient(img_difference, tx, ty, tz, qx, qy, qz, qw)

    #     # DO UPDATE 
    #     learning_rate = 0.01
    #     tx += learning_rate * gradient_tx
    #     ty += learning_rate * gradient_ty
    #     tz += learning_rate * gradient_tz

    #     qx += learning_rate * gradient_qx
    #     qy += learning_rate * gradient_qy
    #     qz += learning_rate * gradient_qz
    #     qw += learning_rate * gradient_qw


    #     if tx < CONVERGENCE_CRITERIUM and ty < CONVERGENCE_CRITERIUM and tz < CONVERGENCE_CRITERIUM and qx < CONVERGENCE_CRITERIUM and qy < CONVERGENCE_CRITERIUM and qz < CONVERGENCE_CRITERIUM and qw < CONVERGENCE_CRITERIUM:
    #         is_converged = True

    #     tx = tx.detach().numpy()
    #     ty = ty.detach().numpy()
    #     tz = tz.detach().numpy()

    #     qx = qx.detach().numpy()
    #     qy = qy.detach().numpy()
    #     qz = qz.detach().numpy()
    #     qw = qw.detach().numpy()
        

    #     # Print gradients
    #     print("Gradient of image difference with respect to tx:", gradient_tx.item())
    #     print("Gradient of image difference with respect to ty:", gradient_ty.item())
    #     print("Gradient of image difference with respect to tz:", gradient_tz.item())
    #     print("Gradient of image difference with respect to qx:", gradient_qx.item())
    #     print("Gradient of image difference with respect to qy:", gradient_qy.item())
    #     print("Gradient of image difference with respect to qz:", gradient_qz.item())
    #     print("Gradient of image difference with respect to qw:", gradient_qw.item())

    #     # Display the original and transformed images
    #     cv2.imshow('Image 1', image1_tensor.detach().squeeze().numpy())
    #     cv2.imshow('Image 2', image2_tensor.detach().squeeze().numpy())
    #     cv2.imshow('Difference Image', ((image1_tensor - image2_tensor).detach().squeeze().numpy() * 255).astype(np.uint8))
    #     cv2.waitKey(1)

    #     # Update the first image for the next iteration
    #     self.obj_frame = image2_tensor


    #     return tx, ty, tz, qx, qy, qz, qw, is_converged

if __name__ == '__main__':
    print("Start Gradient Descent")
    rospy.init_node('gradient_descent', anonymous=True)
    gradient_descent = GradientDescent()

    # # Spin to keep the script alive
    try:
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():

            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")


