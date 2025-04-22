import random
import numpy as np
import cv2
import os
from tqdm import tqdm

# Define the directory for the generated dataset
output_dir = 'car_like_gaussian_augmented_data'
os.makedirs(output_dir, exist_ok=True)

# Load the 100 base dummy images created in grayscale
base_images_dir = 'car_like_geometric_dummy_data'  # Path to the directory of the 100 dummy images
base_images = [cv2.imread(os.path.join(base_images_dir, f'car_like_pattern_{i}.png'), cv2.IMREAD_GRAYSCALE) 
               for i in range(100)]

# Parameters for Gaussian sampling
num_samples = 2000  # Total number of new images to generate
image_size = (32, 32, 3)  # CIFAR-10 image size in RGB
std_dev = 0.5  # Standard deviation for Gaussian noise

# Generate and save images
for i in tqdm(range(num_samples)):
    # Randomly select one of the 100 base images
    base_image = random.choice(base_images)
    
    # Expand grayscale to RGB by stacking along the color channels
    base_image_rgb = np.stack([base_image] * 3, axis=-1)
    
    # Apply isotropic Gaussian noise with mean as the base image pixel value and std deviation 0.5
    noisy_image = base_image_rgb + np.random.normal(0, std_dev, image_size) * 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # Clip values to valid range and convert to uint8
    
    # Save the generated image
    filename = os.path.join(output_dir, f'gaussian_augmented_{i}.png')
    cv2.imwrite(filename, noisy_image)

print(f"Images saved to directory: {output_dir}")
