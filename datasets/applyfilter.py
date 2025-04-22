import numpy as np
import matplotlib.pyplot as plt
import os
import random
from skimage import io
from tqdm import tqdm

def apply_filter(img1, img2, idx, out_dir):
    
    # Ensure both images are of size 32x32
    if img1.shape[:2] != (32, 32) or img2.shape[:2] != (32, 32):
        raise ValueError("Both images must have dimensions of 32x32.")
    
    # Split the images into R, G, B channels
    img1_r, img1_g, img1_b = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
    img2_r, img2_g, img2_b = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]
    
    # Function to apply high-pass and low-pass filters to a channel
    def filter_channel(channel1, channel2, hp_filter, lp_filter):
        # Fourier transform of both channels
        f1 = np.fft.fft2(channel1)
        f1_shift = np.fft.fftshift(f1)
        
        f2 = np.fft.fft2(channel2)
        f2_shift = np.fft.fftshift(f2)
        
        # Apply high-pass filter to the first channel
        f1_hp = f1_shift * lp_filter / 255
        
        # Apply low-pass filter to the second channel
        f2_lp = f2_shift * hp_filter / 255
        
        # Combine the filtered channels in Fourier domain
        # combined = f1_hp + f2_lp
        
        # Inverse Fourier transform to get the combined channel in spatial domain
        # combined_ishift = np.fft.ifftshift(combined)
        # combined_channel = np.fft.ifft2(np.fft.ifftshift(f1_hp))
        combined_channel = np.abs(np.fft.ifft2(np.fft.ifftshift(f1_hp)) + np.fft.ifft2(np.fft.ifftshift(f2_lp)))
        
        return combined_channel

    # Generate high-pass and low-pass filter and replace the filters above
    x = np.linspace(-1, 1, img1_r.shape[0])
    y = np.linspace(-1, 1, img1_r.shape[1])
    X, Y = np.meshgrid(x, y)

    ## Generate 1001 * 1001 matrix that each value represents
    ## distance from the middle
    z = np.sqrt(X**2 + Y**2)

    circle_radius = 0.25
    hp_filter = np.copy(z)
    hp_filter[z > circle_radius] = 255
    hp_filter[z <= circle_radius] = 0

    lp_filter = np.copy(z)
    lp_filter[z > circle_radius] = 0
    lp_filter[z <= circle_radius] = 255
    
    # Apply filters to each channel
    combined_r = filter_channel(img1_r, img2_r, hp_filter, lp_filter)
    combined_g = filter_channel(img1_g, img2_g, hp_filter, lp_filter)
    combined_b = filter_channel(img1_b, img2_b, hp_filter, lp_filter)
    
    # Merge the combined channels back into an RGB image
    combined_img = np.stack((combined_r, combined_g, combined_b), axis=-1)

    # Save the generated image
    io.imsave(f"{out_dir}/gm_{idx+1:05d}.png", np.clip(combined_img, a_min=0, a_max=255.).astype(np.uint8))

# Retrieve 1000 random images from the folder "cifar_train"
def get_random_images(folder_path, num_images=2000):
    # Get list of all .png files in the folder
    all_images = [f for f in os.listdir(folder_path)]
    
    # Randomly select num_images from the list
    # random.shuffle(all_images)

    # Load the selected images using skimage.io
    images = [io.imread(os.path.join(folder_path, img)) for img in all_images]
    
    return images

# Example usage
folder_path = 'cifar_gm_6k'  # Path to the folder containing the images
images = get_random_images(folder_path)
print(len(images))

# Step 1: Generate a set of 100 fixed images with independent and uniform random pixel values between 0 and 1
num_fixed_images = 100
image_resolution = (32, 32, 3)  # RGB images
fixed_images = [np.random.uniform(0, 1, image_resolution) for _ in range(num_fixed_images)]
std_dev = 0.5

# Apply the filter to each image in the list and combine with the white image
for idx, img in tqdm(enumerate(images)):
    # Guassian mixture
    guassian = fixed_images[np.random.randint(0, num_fixed_images)]
    sampled_image = np.random.normal(loc=guassian, scale=std_dev, size=image_resolution) * 255


    apply_filter(img, np.clip(sampled_image, a_min=0, a_max=255.), idx, "gmcifar_6k")
