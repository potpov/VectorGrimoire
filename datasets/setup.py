import os
import shutil
import struct
import sys

from array import array
from os import path
import os
import random
from PIL import Image
from tqdm import tqdm

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

# Function to concatenate and square the images
def concatenate_images(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Determine the target width and height
    target_width = width1 + width2
    target_height = max(height1, height2) * 2

    # Create a new blank image with the target dimensions
    new_image = Image.new("L", (target_width, target_height), color=0)

    # Paste the first image
    new_image.paste(image1, (0, target_height//4))

    # Calculate the coordinates to paste the second image
    paste_x = target_width - width2

    # Paste the second image 
    new_image.paste(image2, (paste_x, target_height//4))

    return new_image

# Randomly select two images and concatenate them
def create_concatenated_image(dataset_path):
    # Get a list of all subdirectories (class labels)
    class_labels = os.listdir(dataset_path)

    # Randomly select two different class labels
    class_label1, class_label2 = random.sample(class_labels, 2)

    # Get the paths to two random image files within each class label
    image1_path = random.choice(os.listdir(os.path.join(dataset_path, class_label1)))
    image2_path = random.choice(os.listdir(os.path.join(dataset_path, class_label2)))

    # Open the images using PIL
    image1 = Image.open(os.path.join(dataset_path, class_label1, image1_path))
    image2 = Image.open(os.path.join(dataset_path, class_label2, image2_path))

    # Concatenate and square the images
    concatenated_image = concatenate_images(image1, image2)

    return concatenated_image

# Create a new dataset of concatenated images
def create_concatenated_dataset(dataset_path, output_path, num_images):
    os.makedirs(output_path, exist_ok=True)

    for i in tqdm(range(num_images)):
        concatenated_image = create_concatenated_image(dataset_path)
        save_path = os.path.join(output_path, f"{i}.png")
        concatenated_image.save(save_path)

        # print(f"Saved concatenated image {i+1}/{num_images}")

def make_mnist_pp(num_concatenated_images_train:int = 50000, num_concatenated_images_test:int=10000):

    # Remove existing dataset
    if(os.path.exists("MNISTpp")):
        shutil.rmtree("MNISTpp")

    # Set the paths to the MNIST dataset folders
    base_path = "datasets/mnist_png"
    train_path = os.path.join(base_path, "training")
    test_path = os.path.join(base_path, "testing")

    # Specify the paths and number of concatenated images to create
    output_dataset_path = "datasets/MNISTpp"

    # Create the concatenated dataset
    print("Creating training dataset")
    create_concatenated_dataset(train_path, os.path.join(output_dataset_path, "training"), num_concatenated_images_train)
    print("Creating testing dataset")
    create_concatenated_dataset(test_path, os.path.join(output_dataset_path, "testing"), num_concatenated_images_test)

if(__name__=="__main__"):
    random.seed(42)

    # Create directories
    create_directory_if_not_exists("datasets/icons8")
    create_directory_if_not_exists("datasets/emojis")

    # Create MNIST++ dataset
    make_mnist_pp()

