import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from matplotlib.patches import Polygon

# This script tests the YOLO model on the real-world dataset test

def load_images_and_labels():
    test_dir = "object-detection/dataset-segm/test"
    image_dir = Path(test_dir) / "images"
    label_dir = Path(test_dir) / "labels"

    image_paths = sorted(list(image_dir.glob('*.jpg')))
    label_paths = sorted(list(label_dir.glob('*.txt')))

    images = [cv2.imread(str(img_path)) for img_path in image_paths]
    labels = []
    
    for lbl_path in label_paths:
        with open(lbl_path, 'r') as file:
            polygons = []
            for line in file:
                points = list(map(float, line.strip().split()[1:]))
                polygon = np.array(points).reshape(-1, 2)
                polygons.append(polygon)
            labels.append(polygons)
    
    return images, labels

def plot_results(image, gt_mask, pred_mask):
    plt.figure(figsize=(15, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Ground Truth Mask
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(gt_mask, cmap='gray')
    plt.axis('off')

    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')

    plt.show()

def convert_yolo_to_pixel_coordinates(polygon, image_shape):
    img_height, img_width = image_shape[:2]
    pixel_coords = np.zeros_like(polygon)
    pixel_coords[:, 0] = polygon[:, 0] * img_width
    pixel_coords[:, 1] = polygon[:, 1] * img_height
    return pixel_coords.astype(np.int32)

def create_mask_from_polygons(image_shape, polygons):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], color=255)
    return mask

def evaluate_model(images, labels):
    for img, gt_polygons in tqdm(zip(images, labels), total=len(images)):
        rgb_image_array = np.asarray(img)

        fig, axes = plt.subplots(1, 5, figsize=(20, 5))

        # Original Image
        axes[0].imshow(cv2.cvtColor(rgb_image_array, cv2.COLOR_BGR2RGB))
        axes[0].set_title("RGB Image")
        axes[0].axis('off')

        # Process ground truth polygons
        if gt_polygons:
            for i, gt_polygon in enumerate(gt_polygons):
                gt_polygon_pixels = convert_yolo_to_pixel_coordinates(gt_polygon, rgb_image_array.shape)
                gt_polygon_mask = create_mask_from_polygons(rgb_image_array.shape, gt_polygon_pixels)
                gt_polygon_patch = Polygon(gt_polygon_pixels[0], closed=True, edgecolor='g', linewidth=2, fill=False)

                axes[i + 1].imshow(gt_polygon_mask, cmap='gray')
                axes[i + 1].add_patch(gt_polygon_patch)
                axes[i + 1].set_title(f"Ground Truth Plate {i+1}")
                axes[i + 1].axis('off')

        # Process YOLO predictions (dummy code, replace with actual YOLO predictions)
        num_predictions = 3  # Replace with actual number of YOLO predictions
        for i in range(num_predictions):
            # Dummy example to show predicted mask and polygon patch
            # Replace this with actual YOLO predictions
            if i == 0:
                # Example prediction
                polygon_xy = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
            else:
                # Another example prediction
                polygon_xy = np.array([[300, 300], [400, 300], [400, 400], [300, 400]])

            polygon_patch = Polygon(polygon_xy, closed=True, edgecolor='r', linewidth=2, fill=False)
            polygon_mask = np.zeros_like(rgb_image_array[:, :, 0], dtype=np.uint8)
            polygon_mask = cv2.fillPoly(polygon_mask, [polygon_xy.astype(np.int32)], color=255)

            axes[i + 3].imshow(polygon_mask, cmap='gray')
            axes[i + 3].add_patch(polygon_patch)
            axes[i + 3].set_title(f"Predicted Plate {i+1}")
            axes[i + 3].axis('off')

        plt.tight_layout()
        plt.show()

def create_segmentation_mask(image_shape, polygons):
    # Create an empty mask of the same size as the image
    segmentation_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Iterate over each polygon
    for polygon_xy in polygons:
        # Convert polygon coordinates to int32 format
        polygon_xy = polygon_xy.astype(np.int32)
        
        # Fill the polygon in the mask
        cv2.fillPoly(segmentation_mask, [polygon_xy], color=255)
    
    return segmentation_mask

def calculate_iou(mask1, mask2):
    # Calculate Intersection over Union (IoU) between two binary masks.

    intersection = np.logical_and(mask1, mask2).astype(np.float32).sum()
    union = np.logical_or(mask1, mask2).astype(np.float32).sum()

    iou = intersection / union if union > 0 else 0.0
    return iou

def evaluate_model(images, labels, device='cuda'):
    # MODEL 
    plate_model_segm = YOLO("object-detection/model/best.pt")


    all_jaccard_scores = []

    for img, gt_mask in tqdm(zip(images, labels), total=len(images)):
        rgb_image_array = np.asarray(img)
        h,w = rgb_image_array.shape[:2]
        prediction = plate_model_segm(rgb_image_array)

        segm_masks_arr = []
        

        if prediction[0].masks != None:
            polygons = []
            polygon_patchs = []
            for i in range(prediction[0].masks.shape[0]):
        
                polygon_xy = prediction[0].masks[i].xy[0]

                # Create a polygon patch
                polygon_patch = Polygon(polygon_xy, closed=True, edgecolor='r', linewidth=2, fill=False)
                # Create a binary mask from the polygon
                polygon_mask = np.zeros_like(rgb_image_array[:, :, 0], dtype=np.uint8)
                polygon_mask = cv2.fillPoly(polygon_mask, [polygon_xy.astype(np.int32)], color=255)

                polygons.append(polygon_xy.astype(np.int32))
                polygon_patchs.append(polygon_patch)
    
                # Apply the mask to the original image
                segmented_image = cv2.bitwise_and(rgb_image_array, rgb_image_array, mask=polygon_mask)

                segm_masks_arr.append(segmented_image)
        # Create a polygon patch
        gt_polygons = []
        gt_polygon_patchs = []
        for i in range(len(gt_mask)):
            gt_polygon = convert_yolo_to_pixel_coordinates(np.array(gt_mask[i]), rgb_image_array.shape)
        
            gt_polygon = np.array(gt_polygon)

            gt_polygon_patch = Polygon(gt_polygon, closed=True, edgecolor='r', linewidth=2, fill=False)
            
            gt_polygons.append(gt_polygon)
            gt_polygon_patchs.append(gt_polygon_patch)
            # Create a binary mask from the polygon
            gt_polygon_mask = np.zeros_like(rgb_image_array[:, :, 0], dtype=np.uint8)
            gt_polygon_mask = cv2.fillPoly(gt_polygon_mask, [gt_polygon.astype(np.int32)], color=255)
            gt_segmented_image = cv2.bitwise_and(rgb_image_array, rgb_image_array, mask=gt_polygon_mask)
            for segmn in segm_masks_arr:
                plt.figure(0)
                plt.imshow(rgb_image_array)
                plt.figure(1)
                plt.imshow(segmn)
                plt.figure(2)
                plt.imshow(gt_segmented_image)
                iou = calculate_iou(segmn, gt_segmented_image)
                print("----------------- IOU -----------------")
                print(iou)
                plt.show()

        segmentation_mask = create_segmentation_mask(rgb_image_array.shape, polygons)
        gt_segmentation_mask = create_segmentation_mask(rgb_image_array.shape, gt_polygons)


        # axes[1].imshow(gt_segmentation_mask, cmap='gray')
        # # axes[1].set_title("Ground Truth Segmentation Mask")
        # axes[1].axis('off')


        # axes[2].imshow(cv2.cvtColor(rgb_image_array, cv2.COLOR_BGR2RGB))
        # for i in range(len(gt_polygon_patchs)):
        #     axes[2].add_patch(gt_polygon_patchs[i])
        # # axes[2].set_title("Ground Truth Polygon Patch")
        # axes[2].axis('off')


        # axes[3].imshow(segmentation_mask, cmap='gray')
        # # axes[3].set_title("Predicted Segmentation Mask")
        # axes[3].axis('off')


        # axes[4].imshow(cv2.cvtColor(rgb_image_array, cv2.COLOR_BGR2RGB))
        # for i in range(len(polygon_patchs)):
        #     axes[4].add_patch(polygon_patchs[i])
        # # axes[4].set_title("Predicted Polygon Patch")
        # axes[4].axis('off')



        plt.tight_layout()
        plt.show()




# Load the test images and labels
images, labels = load_images_and_labels()

# Evaluate the model
evaluate_model(images, labels)

