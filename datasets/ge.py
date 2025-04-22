import numpy as np
import cv2
import os
import random

# Define the directory to save the generated images
output_dir = 'car_like_geometric_dummy_data'
os.makedirs(output_dir, exist_ok=True)

# Parameters for the dataset
num_images = 100  # Number of dummy images to generate
image_size = (32, 32)  # Size of each image (similar to CIFAR-10)

# Function to create a car-like pattern with geometric elements
def create_car_like_pattern(size):
    img = np.ones(size, dtype=np.uint8) * 255  # White background
    
    # Randomize car body position and size slightly
    body_width = random.randint(18, 24)
    body_height = random.randint(6, 8)
    car_body_top_left = (random.randint(4, 8), random.randint(14, 18))
    car_body_bottom_right = (car_body_top_left[0] + body_width, car_body_top_left[1] + body_height)
    cv2.rectangle(img, car_body_top_left, car_body_bottom_right, 0, -1)

    # Randomize wheel positions and sizes
    wheel_radius = random.randint(2, 3)
    wheel_offset = random.randint(2, 4)
    cv2.circle(img, (car_body_top_left[0] + wheel_offset, car_body_bottom_right[1] + wheel_radius + 1), wheel_radius, 0, -1)
    cv2.circle(img, (car_body_bottom_right[0] - wheel_offset, car_body_bottom_right[1] + wheel_radius + 1), wheel_radius, 0, -1)

    # Randomize window position and size slightly
    window_width = random.randint(8, 12)
    window_height = random.randint(2, 4)
    window_top_left = (car_body_top_left[0] + 2, car_body_top_left[1] + 1)
    window_bottom_right = (window_top_left[0] + window_width, window_top_left[1] + window_height)
    cv2.rectangle(img, window_top_left, window_bottom_right, 255, -1)  # White window

    # Add a random grill-like pattern (lines) within the car body
    for i in range(window_top_left[0], window_bottom_right[0], random.randint(2, 4)):
        cv2.line(img, (i, car_body_top_left[1]), (i, car_body_bottom_right[1]), 0, 1)

    return img

# Generate and save images with randomness
for i in range(num_images):
    pattern = create_car_like_pattern(image_size)
    filename = os.path.join(output_dir, f'car_like_pattern_{i}.png')
    cv2.imwrite(filename, pattern)

print(f"Images saved to directory: {output_dir}")
