import numpy as np
import os
from skimage import io
from tqdm import tqdm

# Load the .npz file
npz_file_path = 'gmcifar_6k_ema_rate0.9999_100000_4000x32x32x3.npz'  # Change this to your actual file path
output_dir = 'gmcifar_6k_sample'
data = np.load(npz_file_path)

# Extract images and labels
images = data['arr_0']
labels = data['arr_1']

# Create output directory if it doesn't exist

os.makedirs(output_dir, exist_ok=True)

# Iterate through images and save each one
for i in tqdm(range(len(images)), desc='Saving images'):
    image = images[i]
    # label = labels[i]
    image_path = os.path.join(output_dir, f'image_{i}.png')
    io.imsave(image_path, image)

print(f'Images successfully extracted to "{output_dir}"')
