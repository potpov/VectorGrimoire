import random

import pandas as pd
import numpy as np
from svgpathtools import svg2paths, svg2paths2, disvg, Path  # this is used to READ and breakdown SVG
from svgwrite import Drawing
from cairosvg import svg2png
from PIL import Image
from matplotlib import pyplot as plt
import os
import io
from tqdm import tqdm
from pathlib import Path as LinuxPath
from glob import glob
import argparse
import torch
import seaborn as sns
import math

OUT_W = 128
OUT_H = 128

def raster(svg_file: Drawing):
    """
    This function simply resizes and rasters a series of Paths
    @param svg_file: Drawing object
    @return: Numpy array of the raster image
    """
    svg_png_image = svg2png(
        bytestring=svg_file.tostring(),
        output_width=OUT_W,
        output_height=OUT_H)
    img = Image.open(io.BytesIO(svg_png_image))
    # convert to numpy array
    img = np.array(img)
    # img = np.flip(img, axis=0)  # images are rastered upside down -> wtf?
    # img = np.flip(img, axis=0)  # images are rastered upside down -> wtf?
    img = 255 - img[:, :, 3]  # RGBA -> grey-scale
    return img.astype(np.uint8)

def plot_segments(rasterized_segments, title:str="A disassembled tree"):
    nrows = math.ceil(len(rasterized_segments) / 8)
    ncols = 8
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(5*ncols, 5*nrows))
    for i, img in enumerate(rasterized_segments):
        curr_row = i // ncols
        curr_col = i % ncols
        axs[curr_row][curr_col].imshow(img, cmap="gray")
        axs[curr_row][curr_col].axis("off")
    if title is not None:
        axs[0][ncols//2].set_title(title)

def plot_merged_segments(rasterized_segments, title=None):
    plt.imshow(np.array(rasterized_segments).min(axis=0), cmap="gray")

def get_flattened_paths(paths):
    flattened_paths = [segment for path in paths for segment in path._segments]
    return flattened_paths

def get_single_paths(paths, filter_zero_length = True):
    flattened_paths = get_flattened_paths(paths)
    single_paths = [Path(element) for element in flattened_paths]
    if filter_zero_length:
        single_paths = [path for path in single_paths if path.length() > 0.]
        
    return single_paths

def get_rasterized_segments(single_paths:list, stroke_width:float, svg_attributes, centered = False):
    if centered:
        viewbox = None
    else:
        viewbox=svg_attributes["viewBox"]
    return np.array([raster(disvg(my_path, paths2Drawing=True, stroke_widths=[stroke_width] * len(my_path), viewbox=viewbox)) for my_path in single_paths if my_path.length() > 0.])

def svg_path_to_segment_image_arrays(svg_path):
    """
    This function takes a path to an SVG file and returns two numpy arrays of the rasterized path segments.

    Inputs:
        svg_path: path to the SVG file
    
    Returns:
        rasterized_segments_centered: numpy array of the rasterized segments, all placed in the middle of the image
        rasterized_segments: numpy array of the rasterized segments, placed on their relative position where they belong
    """
    paths, attributes, svg_attributes = svg2paths2(svg_path)
    single_paths = get_single_paths(paths)

    # everything placed in the middle
    rasterized_segments_centered = get_rasterized_segments(single_paths, stroke_width = 0.5, svg_attributes=svg_attributes, centered=True)

    # everything placed where it belongs
    rasterized_segments = get_rasterized_segments(single_paths, stroke_width = 2.0, svg_attributes=svg_attributes, centered=False)

    return rasterized_segments_centered, rasterized_segments

def get_positional_array_from_paths(single_paths, svg_attributes):
    viewbox_x, viewbox_y, viewbox_w, viewbox_h = [float(x) for x in svg_attributes["viewBox"].split(" ")]
    assert viewbox_x == 0 and viewbox_y == 0, "you require normalization of viewbox"
    abs_start_points = []
    abs_end_points = []
    rel_start_points = []
    rel_end_points = []
    for i, path in enumerate(single_paths):
        abs_start_points.append([path.start.real, path.start.imag])
        abs_end_points.append([path.end.real, path.end.imag])

        rel_start_x = path.start.real / viewbox_w
        rel_start_y = path.start.imag / viewbox_h

        rel_start_points.append([rel_start_x, rel_start_y])

        rel_end_x = path.end.real / viewbox_w
        rel_end_y = path.end.imag / viewbox_h

        rel_end_points.append([rel_end_x, rel_end_y])
    
    stacked_points = np.stack([abs_start_points, abs_end_points,  rel_start_points,  rel_end_points], axis=1)
    return stacked_points 

if __name__ == "__main__":
    SEGMENT_THRESHOLD = 100  # equivalent to context length
    CLASS = "tree"
    FIGR8_PATH = "/scratch2/moritz_data/figr8_trees_processed/"
    OUT_DIR = f"/scratch2/moritz_data/causal_figr8_trees_T{SEGMENT_THRESHOLD}"
    OVERRIDE = True

    if OVERRIDE:
        if os.path.exists(OUT_DIR):
            input("You are about to override the output directory. Press any key to continue...")
            os.system(f"rm -rf {OUT_DIR}")

    if not os.path.exists(OUT_DIR):
        os.makedirs(os.path.join(OUT_DIR, CLASS))

    df = pd.DataFrame(columns=['filename', 'class', 'split'])
    statistic_df = pd.read_csv("/scratch2/moritz_data/figr8_trees_statistics.csv")
    all_paths = statistic_df[statistic_df["num_segments"] < SEGMENT_THRESHOLD].file.values

    for i, path in enumerate(tqdm(all_paths)):
        paths, attributes, svg_attributes = svg2paths2(path)
        single_paths = get_single_paths(paths)

        rasterized_segments_centered = get_rasterized_segments(single_paths, stroke_width = 0.5, svg_attributes=svg_attributes, centered=True)
        rasterized_segments = get_rasterized_segments(single_paths, stroke_width = 2.0, svg_attributes=svg_attributes, centered=False)

        position_information = get_positional_array_from_paths(single_paths, svg_attributes)

        assert position_information.shape[0] == rasterized_segments_centered.shape[0] == rasterized_segments.shape[0], "something went wrong"

        raster_filename_absolute = f"I{i}_{len(rasterized_segments)}_Segments_images_absolute.npy"
        raster_filename_centered = f"I{i}_{len(rasterized_segments)}_Segments_images_centered.npy"
        position_filename = f"I{i}_{len(rasterized_segments)}_Segments_positions.npy"

        np.save(os.path.join(OUT_DIR, CLASS, raster_filename_absolute), rasterized_segments)
        np.save(os.path.join(OUT_DIR, CLASS, raster_filename_centered), rasterized_segments_centered)
        np.save(os.path.join(OUT_DIR, CLASS, position_filename), position_information)
        
        new_row = {
            "raster_filename_absolute": raster_filename_absolute,
            "raster_filename_centered": raster_filename_centered,
            "position_filename": position_filename,
            "class": CLASS,
            "split": np.random.choice(["train", "test"], p=[0.8, 0.2])
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(os.path.join(OUT_DIR, CLASS, 'split.csv'), index=False)