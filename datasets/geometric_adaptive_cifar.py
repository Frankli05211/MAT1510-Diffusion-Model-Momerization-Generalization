import numpy as np
import cv2
import os
import random
from tqdm import tqdm

# Define the directory to save the generated images
output_dir = 'enhanced_1k'
os.makedirs(output_dir, exist_ok=True)

# Parameters for the dataset
num_images = 1000  # Number of dummy images to generate
image_size = (32, 32, 3)  # CIFAR-10 image size in RGB

# Function to create enhanced geometric adaptive dummy data
def create_enhanced_geometric_pattern(size):
    img = np.ones(size, dtype=np.uint8) * 255  # White background in RGB

    # Random choice of pattern type: grid, lines, circles, overlapping shapes, gradients
    pattern_type = random.choice(['grid', 'lines', 'circles', 'overlapping_shapes', 'gradient'])

    if pattern_type == 'grid':
        # Create a multi-colored grid pattern
        step = random.randint(4, 8)  # Random grid step size
        for i in range(0, size[0], step):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.line(img, (i, 0), (i, size[1] - 1), color, 1)
            cv2.line(img, (0, i), (size[0] - 1, i), color, 1)

    elif pattern_type == 'lines':
        # Create horizontal or vertical lines with random colors
        direction = random.choice(['horizontal', 'vertical'])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if direction == 'horizontal':
            for i in range(0, size[0], random.randint(4, 8)):
                cv2.line(img, (0, i), (size[1] - 1, i), color, 1)
        else:
            for i in range(0, size[1], random.randint(4, 8)):
                cv2.line(img, (i, 0), (i, size[0] - 1), color, 1)

    elif pattern_type == 'circles':
        # Create circles with random colors and varying sizes
        num_circles = random.randint(3, 6)  # Random number of circles
        for _ in range(num_circles):
            center = (random.randint(0, size[1] - 1), random.randint(0, size[0] - 1))
            radius = random.randint(3, 6)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.circle(img, center, radius, color, -1)

    elif pattern_type == 'overlapping_shapes':
        # Create overlapping circles and rectangles with random colors
        num_shapes = random.randint(4, 8)  # Random number of shapes
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rectangle'])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if shape_type == 'circle':
                center = (random.randint(0, size[1] - 1), random.randint(0, size[0] - 1))
                radius = random.randint(3, 8)
                cv2.circle(img, center, radius, color, -1)
            elif shape_type == 'rectangle':
                top_left = (random.randint(0, size[1] // 2), random.randint(0, size[0] // 2))
                bottom_right = (top_left[0] + random.randint(4, 10), top_left[1] + random.randint(4, 10))
                cv2.rectangle(img, top_left, bottom_right, color, -1)

    elif pattern_type == 'gradient':
        # Create a gradient pattern
        start_color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.float32)
        end_color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.float32)
        for y in range(size[0]):
            alpha = y / size[0]
            color = (1 - alpha) * start_color + alpha * end_color
            color = tuple(map(int, color))
            cv2.line(img, (0, y), (size[1] - 1, y), color, 1)

    return img

# Generate and save images with randomness and enhanced complexity
for i in tqdm(range(num_images)):
    pattern = create_enhanced_geometric_pattern(image_size)
    filename = os.path.join(output_dir, f'enhanced_{i:05d}.png')
    cv2.imwrite(filename, pattern)

print(f"Images saved to directory: {output_dir}")
