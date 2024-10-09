import os
import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.metrics import pairwise_distances
from torchvision.transforms import CenterCrop

folder_path = "/raid/marco.cipriano/data/SVG/Grimoire/Cartoons/raw"
output_path = "/raid/marco.cipriano/data/SVG/Grimoire/Cartoons/preprocessed"

device = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = False
sam_checkpoint = "/raid/marco.cipriano/weights/sam_vit_h_4b8939.pth"
model_type = "default"
palette_size = 4096
min_threshold = 0.05
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def merge_layers(masks, colors):
    """
    Reconstruct the image using masks and colors.
    """

    binary_masks = [m["segmentation"] for m in masks]
    result = np.full((*binary_masks[0].shape, 3), 1.)
    # Assign each mask a label and mark overlaps in the hierarchy
    for idx, mask in enumerate(binary_masks):
        result[mask] = colors[idx]
    return result


def generate_palette(num_colors=64):
    """
    Generate a palette of `num_colors` evenly distributed RGB triplets.

    Parameters:
    - num_colors: Number of colors to generate (default: 64)

    Returns:
    - palette: Array of shape (num_colors, 3) containing RGB values in range [0, 255]
    """

    # Number of steps for each channel to get approximately `num_colors` total combinations
    steps = int(np.ceil(num_colors ** (1 / 3)))  # Cube root to get similar number of steps for R, G, B
    assert steps <= 256, "Too many steps for a basic palette"

    values = np.linspace(0, 255, steps, dtype=int)
    palette = np.array(np.meshgrid(values, values, values)).T.reshape(-1, 3)
    return palette


def map_to_palette(image, palette):
    """
    Map each pixel in the image to the closest color in the palette.

    Parameters:
    - image: Input image as an (H, W, 3) array in range [0, 255]
    - palette: Array of shape (N, 3) containing N RGB triplets

    Returns:
    - indexed_image: Array of shape (H, W) containing indices of the closest colors in the palette
    """
    h, w, _ = image.shape
    reshaped_image = image.reshape(-1, 3)

    # Calculate distances between each pixel and each color in the palette
    distances = pairwise_distances(reshaped_image, palette)

    # Find the index of the closest color for each pixel
    closest_color_indices = np.argmin(distances, axis=1)

    # Reshape the result back to the original image shape (H, W)
    indexed_image = closest_color_indices.reshape(h, w)

    return indexed_image


def indexed_to_rgb(indexed_image, palette):
    """
    Convert an indexed image (with N color indices) back to an RGB image using the palette.

    Parameters:
    - indexed_image: Array of shape (H, W) containing indices in range [0, N-1]
    - palette: Array of shape (N, 3) containing N RGB triplets

    Returns:
    - rgb_image: Array of shape (H, W, 3) containing RGB values
    """
    # Map each index to its corresponding color in the palette
    rgb_image = palette[indexed_image]

    return rgb_image


def process_folder(folder_path):
    """
    Processes all images in the folder and creates a mask hierarchy for each.
    """
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    palette = generate_palette(palette_size)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        # min_mask_region_area=int(0.0005 * h * w)
    )

    for filename in tqdm(filenames, total=len(filenames)):
        image_path = os.path.join(folder_path, filename)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # pre- processing
        crop = CenterCrop((400, 400))
        image = crop(torch.from_numpy(image).moveaxis(-1, 0))
        image = image.moveaxis(0, -1).numpy()

        masks = mask_generator.generate(image)
        # move everything within [0, N] values
        q_image = map_to_palette(image, palette)

        # re-order masks -> Background at the bottom
        hierarcy = np.argsort([m["area"] for m in masks])[::-1]

        if DEBUG:
            plt.imshow(image)
            plt.show()

        h, w, _ = image.shape
        tot_area = h * w
        # masks = [masks[i] for i in hierarcy if (100 * masks[i]["area"] / tot_area) > min_threshold]  # reorder mask according to their area
        masks = [masks[i] for i in hierarcy]  # reorder mask according to their area

        color_masks = []
        colors = []
        for mask in masks:

            mask_values = q_image[mask["segmentation"]]
            # color_idx = int(np.median(mask_values).item())
            bins = np.bincount(mask_values).argsort()
            # small rule to reward wait if it's the second most common value
            color_idx = int(bins[-2]) if bins[-2] == (len(palette) - 1) else int(bins[-1])
            # color_idx = int(np.bincount(mask_values).argmax())
            assert len(palette) > color_idx
            target_color = palette[color_idx] / 255
            colors.append(target_color)

            assert np.all(target_color <= 1.)
            target_image = np.full_like(image, 1., dtype=np.float32)
            target_image[mask["segmentation"]] = target_color
            color_masks.append(target_image)

        if DEBUG:
            _ = merge_layers(masks, colors)
            plt.imshow(_)
            plt.show()

        is_train = np.random.rand() < 0.8
        split_folder = "train" if is_train else "val"
        Path(os.path.join(output_path, "color_masks", split_folder)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output_path, "binary_masks", "colors", split_folder)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output_path, "binary_masks", "masks", split_folder)).mkdir(parents=True, exist_ok=True)

        save_name = filename.replace(".png", ".npy")
        np.save(os.path.join(output_path, "color_masks", split_folder, save_name), np.stack(color_masks))
        np.save(os.path.join(output_path, "binary_masks", "colors", split_folder, save_name), np.stack(colors))
        np.save(os.path.join(output_path, "binary_masks", "masks", split_folder, save_name), np.stack([m["segmentation"] for m in masks]))


if __name__ == "__main__":
    process_folder(folder_path)
