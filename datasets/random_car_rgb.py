import random
import numpy as np
import cv2
import os

# Define the directory to save the generated images
output_dir = 'random_car_like_rgb_dummy_data'
os.makedirs(output_dir, exist_ok=True)

# Parameters for the dataset
num_images = 4000  # Number of dummy images to generate
image_size = (32, 32, 3)  # CIFAR-10 image size in RGB

# Function to create a car-like RGB pattern with geometric elements and randomness
def create_random_car_like_pattern(size):
    img = np.ones(size, dtype=np.uint8) * 255  # White background in RGB

    # Randomize car body color and position
    car_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))  # Random RGB color
    body_width = random.randint(18, 24)
    body_height = random.randint(6, 8)
    car_body_top_left = (random.randint(4, 8), random.randint(14, 18))
    car_body_bottom_right = (car_body_top_left[0] + body_width, car_body_top_left[1] + body_height)
    cv2.rectangle(img, car_body_top_left, car_body_bottom_right, car_color, -1)

    # Randomize wheels color, positions, and sizes
    wheel_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))  # Darker color for wheels
    wheel_radius = random.randint(2, 3)
    wheel_offset = random.randint(2, 4)
    cv2.circle(img, (car_body_top_left[0] + wheel_offset, car_body_bottom_right[1] + wheel_radius + 1), wheel_radius, wheel_color, -1)
    cv2.circle(img, (car_body_bottom_right[0] - wheel_offset, car_body_bottom_right[1] + wheel_radius + 1), wheel_radius, wheel_color, -1)

    # Randomize window color, position, and size slightly
    window_color = (255, 255, 255)  # White for window
    window_width = random.randint(8, 12)
    window_height = random.randint(2, 4)
    window_top_left = (car_body_top_left[0] + 2, car_body_top_left[1] + 1)
    window_bottom_right = (window_top_left[0] + window_width, window_top_left[1] + window_height)
    cv2.rectangle(img, window_top_left, window_bottom_right, window_color, -1)

    # Add a random grill-like pattern (lines) within the car body
    grill_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))  # Dark lines for grill
    for i in range(window_top_left[0], window_bottom_right[0], random.randint(2, 4)):
        cv2.line(img, (i, car_body_top_left[1]), (i, car_body_bottom_right[1]), grill_color, 1)

    return img

# Generate and save images with randomness
for i in range(num_images):
    pattern = create_random_car_like_pattern(image_size)
    filename = os.path.join(output_dir, f'dummy_{i}.png')
    cv2.imwrite(filename, pattern)

print(f"Images saved to directory: {output_dir}")
