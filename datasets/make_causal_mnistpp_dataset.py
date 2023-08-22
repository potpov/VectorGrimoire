"""
This script generates the CausalMNIST++ dataset.
A single image of the dataset is a horizontal concatenation of `random.randint(1, context_length - 1)` MNIST digits with the condition that they must be in ascending order from left to right. 
Currently, the images also always start on the very left with a zero digit. 
This constraint could be lifted for a more "difficult" dataset, e.g. start with a random digit but keep the ascending order constraint.
Note: As the output resolution is fixed, the digits get smaller the more there are.

The images are saved as numpy arrays in the following format:
    - shape: (num_timesteps, target_resolution, target_resolution)
    - background: 255 (white)
    - digits: 0 (black)
    - dtype: uint8

num_timesteps varies between 1 and 9 (inclusive) and is determined randomly for each image.
The train/test split is 0.75/0.25 and a split.csv file with more information is created in the output directory.
"""

import argparse
import shutil
import os
import random
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
import pandas as pd


def concatenate_images_to_np_array(unified_images: list, target_resolution:int = 128):
    """
    Takes a list of square PIL Images and reshapes and repositions them to be a perfect horizontal concatenation if they were added up.
    Does not actually add them up, but returns a numpy array of all the individual shape layers.

    @param unified_images: a list of square PIL Images
    @param target_resolution: the resolution of the output images
    @return: a numpy array of shape (len(unified_images), target_resolution, target_resolution) containing the images
    """
    width, height = unified_images[0].size

    target_width = len(unified_images) * width
    target_height = target_width

    step_size = target_width // len(unified_images)
    new_images = []
    for i, image in enumerate(unified_images):
        new_image = Image.new("L", (target_width, target_height), color="white")
        new_x = step_size*i
        new_y = target_height//2 - height//2
        new_image.paste(image, (new_x, new_y))
        new_images.append(np.array(new_image.resize((target_resolution, target_resolution))))

    return np.array(new_images)

def sample_digit(base_path:str, label:str, split:str = "training"):
    """
    Samples a single digit from the MNIST dataset.

    @param base_path: the base path to the MNIST dataset as image files with black bg and white digit
    @param label: the label of the digit to sample
    @param split: the split to sample from (either "training" or "testing")
    @return: a PIL image of the sampled digit (already inverted, so white bg & black digit)
    """
    assert label in "0123456789", "please provide a single digit as label"
    assert split in ["training", "testing"], "please provide a valid split"
    path = os.path.join(base_path, split, label)
    return ImageOps.invert(Image.open(os.path.join(path, random.choice(os.listdir(path)))))  # inverting is required as the base images are white on black

def make_causal_mnist_pp_dataset(base_path:str, num_samples:int, output_path:str, context_length:int = 10, seed:int = 42):
    """
    Creates a dataset of causal MNIST++ images.

    @param base_path: the base path to the MNIST dataset
    @param num_samples: the number of samples to create
    @return: a numpy array of shape (num_samples, 28, 28) containing the sampled images
    """
    assert os.path.exists(base_path), "please provide a valid path to the MNIST dataset"
    assert context_length <= 10, "context length must be <= 10 as there are only 10 digits"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "train"))
    os.makedirs(os.path.join(output_path, "test"))
    random.seed(seed)
    np.random.seed(seed)

    df = pd.DataFrame(columns=['filename', 'class', 'split'])

    for i in tqdm(range(num_samples)):
        num_features = random.randint(1, context_length-1)  # context length minus one because we add white image in dataloader
        digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"][:num_features]
        sample_images = concatenate_images_to_np_array([sample_digit(base_path, digit) for digit in digits])
        split = np.random.choice(["train", "test"], p=[0.75, 0.25])

        # TODO if we start with a random digit, this must be changed to include the start digit in the filename
        filename = f"I{i}_P{num_features}.npy"
        np.save(os.path.join(output_path, split, filename), sample_images)
        new_row = {
                "filename": filename,
                "class": num_features,
                "split": split
            }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(os.path.join(output_path, 'split.csv'), index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Export for SVG dataset')
    parser.add_argument("--input_path", "-i", help="path to the MNIST dataset", default="/home/mfeuerpfeil/master/thesis/datasets/mnist_png")
    parser.add_argument('--context_len', '-l', help='max sub-images, must be less than 10', default=10)
    parser.add_argument("--num_samples", "-n", help="number of samples to create, 0.72/0.25 train/test split", default=15000)
    parser.add_argument("--output_path", "-o", help="path to the output directory", default="/scratch2/CausalMNISTpp")
    args = parser.parse_args()

    make_causal_mnist_pp_dataset(args.input_path,
                                 int(args.num_samples),
                                 args.output_path,
                                 context_length=int(args.context_len),
                                 seed=42)
