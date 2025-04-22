import numpy as np
import os
from skimage import io
from tqdm import tqdm

# Function to calculate L2 norm between two images
def l2_norm(img1, img2):
    return np.linalg.norm(img1 - img2)

# Function to evaluate replicates
def evaluate_replicates(train_folder, *sample_folders):
    # Load all train images
    train_images = []
    for train_file in tqdm(os.listdir(train_folder), desc='Loading train images'):
        train_image_path = os.path.join(train_folder, train_file)
        train_image = io.imread(train_image_path).astype(float) / 255
        train_images.append(train_image)
    train_images = np.array(train_images)

    # Iterate through all sample folders
    for sample_folder in sample_folders:
        replicate_count = 0
        sample_images = []

        # Load all sample images
        for sample_file in tqdm(os.listdir(sample_folder), desc=f'Loading sample images from {sample_folder}'):
            sample_image_path = os.path.join(sample_folder, sample_file)
            sample_image = io.imread(sample_image_path).astype(float) / 255
            sample_images.append(sample_image)

        # Compare each sample image to all train images
        for sample_image in tqdm(sample_images, desc=f'Comparing images for {sample_folder}'):
            l2_distances = [l2_norm(sample_image, train_image) for train_image in train_images]
            l2_distances.sort()

            # Calculate the ratio of smallest to second smallest L2 norm
            if len(l2_distances) >= 2:
                ratio = l2_distances[0] / l2_distances[1]
                if ratio < 1 / 3:
                    replicate_count += 1

        # Report the proportion of replicates
        total_samples = len(sample_images)
        proportion = replicate_count / total_samples if total_samples > 0 else 0
        print(f'Proportion of replicates in {sample_folder}: {proportion:.2f}')
        print(f'Number of replicates in {sample_folder}: {replicate_count}')

# Example usage
train_folder_path = 'datasets/cifar_train'  # Replace with your train folder path
sample_folder_paths = ['enhanced_sample_dir/enhanced_3kmodel_samples', 'enhanced_sample_dir/enhanced_4kmodel_samples', 'enhanced_sample_dir/enhanced_6kmodel_samples']  # Replace with your sample folder paths
evaluate_replicates(train_folder_path, *sample_folder_paths)
