import os
import numpy as np
from PIL import Image

# Directories containing the images and corresponding data
directories_img = [
    './RenderProduct_Replicator/rgb',
    './RenderProduct_Replicator_01/rgb',
    './RenderProduct_Replicator_02/rgb'
]
directories_data = [
    './RenderProduct_Replicator/bounding_box_2d_tight',
    './RenderProduct_Replicator_01/bounding_box_2d_tight',
    './RenderProduct_Replicator_02/bounding_box_2d_tight'
]
output_dir = './dataset_syn/train/'

# Create output directories for images and labels if they don't exist
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

counter = 0
# Iterate over each directory containing images and data
for directory_img, directory_data in zip(directories_img, directories_data):
    # List all files in the directories
    files_img = sorted(os.listdir(directory_img))
    files_data = sorted(os.listdir(directory_data))

    # Filter to only include image files (you may need to adjust the extensions)
    image_files = [file for file in files_img if file.endswith('.png')]

    # Loop through each fifth image file
    for idx, image_file in enumerate(image_files):
        if idx % 4 == 0:
            image_path = os.path.join(directory_img, image_file)
            data_file = os.path.join(directory_data, f'bounding_box_2d_tight_{image_file[-8:-4]}.npy')

            # Load image
            image = Image.open(image_path)
            width, height = image.size

            # Load bounding box data
            bbox_data = np.load(data_file)

            # Create label file
            new_labe_file = f'image_{counter:04d}.txt'
            label_file = os.path.join(output_dir, 'labels', new_labe_file)

            # Write label data to file
            with open(label_file, 'w') as f:
                for bbox in bbox_data:
                    # Convert bounding box coordinates to YOLO format (relative)
                    x_center = (bbox[1] + bbox[3]) / (2 * width)
                    y_center = (bbox[2] + bbox[4]) / (2 * height)
                    box_width = (bbox[3] - bbox[1]) / width
                    box_height = (bbox[4] - bbox[2]) / height

                    # Write label in YOLO format (class x_center y_center width height)
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

            # Rename image file and save to output directory
            new_image_file = f'image_bckgr_{counter:04d}.png'  # Renaming image
            new_image_path = os.path.join(output_dir, 'images', new_image_file)
            image.save(new_image_path)
            print(counter)
            counter +=1 
