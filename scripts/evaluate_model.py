import os
import numpy as np
import skimage.io as io
import argparse
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Function to calculate L2 norm between two images
def l2_norm(img1, img2):
    return np.linalg.norm(img1 - img2)

def main(training_folders, sample_folders, output_folder):
    # Initialize lists to store results for plotting
    sample_counts = []
    replicate_proportions = []

    # Process each sample folder and corresponding training folder
    for training_folder, sample_folder in zip(training_folders, sample_folders):
        # Load all training images into a list
        training_images = []
        training_filenames = []
        for filename in os.listdir(training_folder):
            img_path = os.path.join(training_folder, filename)
            img = io.imread(img_path).astype(float)/255
            training_images.append(img)
            training_filenames.append(filename)

        # Load all sample images into a list
        sample_images = []
        sample_filenames = []
        for filename in os.listdir(sample_folder):
            img_path = os.path.join(sample_folder, filename)
            img = io.imread(img_path).astype(float)/255
            sample_images.append(img)
            sample_filenames.append(filename)

        # Check for replicates
        replicate_count = 0
        replicate_print_count = 0
        nonreplicate_print_count = 0

        for sample_img, sample_filename in tqdm(zip(sample_images, sample_filenames)):
            # Calculate L2 norms between the sample image and all training images
            distances = np.array([l2_norm(sample_img, train_img) for train_img in training_images])
            
            # Find the index of the training image with the smallest L2 norm
            min_index = np.argmin(distances)
            min_distance = distances[min_index]
            
            # Find the smallest and second smallest L2 norms
            sorted_distances = np.sort(distances)
            if len(sorted_distances) >= 2:
                d1, d2 = sorted_distances[0], sorted_distances[1]
                
                # Check the replicate condition
                if d1 / d2 < 1/3:
                    replicate_count += 1
                    if (replicate_print_count < 10):
                        replicate_print_count += 1
                        # Print the sample image name and the corresponding training image name with the smallest L2 norm
                        print(f'Replicate: Sample Image: {sample_filename} -> Closest Training Image: {training_filenames[min_index]} (L2 Norm: {min_distance:.4f})')
                else:
                    if (nonreplicate_print_count < 10):
                        nonreplicate_print_count += 1
                        # Print the sample image name and the corresponding training image name with the smallest L2 norm
                        print(f'Nonreplicate: Sample Image: {sample_filename} -> Closest Training Image: {training_filenames[min_index]} (L2 Norm: {min_distance:.4f})')

        # Calculate proportion of replicates
        total_samples = len(sample_images)
        replicate_proportion = replicate_count / total_samples

        # Print results
        print(f'Sample Folder: {sample_folder} compared with Training Folder: {training_folder}')
        print(f'Number of replicates: {replicate_count}')
        print(f'Proportion of replicates: {replicate_proportion:.4f}')

        # Store results for plotting
        sample_counts.append(total_samples)
        replicate_proportions.append(replicate_proportion)

    # Plot the relationship between sample counts and replicate proportions
    x_values = [2, 3, 4, 6]  # Hardcoded x-axis values
    plt.figure()
    plt.plot(x_values, replicate_proportions, marker='o')
    plt.xlabel('Train Set Size (x1k samples)')
    plt.ylabel('Proportion of Replicates')
    # plt.title('GM ')
    
    # Save the plot to the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plot_path = os.path.join(output_folder, 'new_gmcifar_replicate_proportion_plot.png')
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect replicated images between training and sample sets.")
    parser.add_argument('--training_folders', type=str, nargs=4, required=True, help="Paths to the 4 training folders")
    parser.add_argument('--sample_folders', type=str, nargs=4, required=True, help="Paths to the 4 sample folders")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output folder to save the plot")
    args = parser.parse_args()

    main(args.training_folders, args.sample_folders, args.output_folder)
