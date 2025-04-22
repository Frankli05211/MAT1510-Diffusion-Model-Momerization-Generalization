import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from tqdm import tqdm

# Create a directory to save the images
output_dir = "gm_4k"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Generate a set of 100 fixed images with independent and uniform random pixel values between 0 and 1
num_fixed_images = 100
image_resolution = (32, 32, 3)  # RGB images
fixed_images = [np.random.uniform(0, 1, image_resolution) for _ in range(num_fixed_images)]

# Step 2: Generate 2000 images by sampling from isotropic Gaussian distribution
num_generated_images = 4000
std_dev = 0.5

# Use tqdm to display progress
for i in tqdm(range(num_generated_images)):
    # Randomly select one of the 100 fixed images
    X = fixed_images[np.random.randint(0, num_fixed_images)]
    
    # Sample an image from the isotropic Gaussian distribution with mean X and standard deviation 0.5 for each pixel
    sampled_image = np.random.normal(loc=X, scale=std_dev, size=image_resolution) * 255
    
    # Step 3: Clip the sampled image to take values within [0, 1]
    # clipped_image = np.clip(sampled_image, 0, 1)
    
    # Save the image using matplotlib
    io.imsave(f"{output_dir}/gm_{i+1:05d}.png", np.clip(sampled_image, a_min=0, a_max=255.).astype(np.uint8))

print(f"{num_generated_images} images sampled from the Gaussian distribution have been saved in the '{output_dir}' directory.")
