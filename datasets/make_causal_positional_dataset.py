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

import math
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
    plt.imshow(np.array(rasterized_segments).min(axis=0), cmap="gray", title = title)

def get_flattened_paths(paths):
    flattened_paths = [segment for path in paths for segment in path._segments]
    return flattened_paths

def get_single_paths(paths, filter_zero_length = True):
    flattened_paths = get_flattened_paths(paths)
    single_paths = [Path(element) for element in flattened_paths]
    if filter_zero_length:
        single_paths = [path for path in single_paths if path.length() > 0.]
        
    return single_paths

def calc_max_diff(single_paths):
    total_max_diff = 0
    for idx in range(len(single_paths)):
        abs_start = single_paths[idx].start #- single_paths[0].end
        abs_end = single_paths[idx].end #- single_paths[0].end
        top_left = complex(min(abs_start.real, abs_end.real), min(abs_start.imag, abs_end.imag))
        bottom_right = complex(max(abs_start.real, abs_end.real), max(abs_start.imag, abs_end.imag))
        diff = bottom_right - top_left
        max_diff = max(diff.real, diff.imag)
        if max_diff > total_max_diff:
            total_max_diff = max_diff
    return total_max_diff

def all_paths_to_max_diff(all_paths, index:int = 1):
    """
    calculates the index'th maximum width of a single path in all_paths.
    index is the idx of the max_diff you want to get. idx=0 is largest, idx=1 is second largest, etc.
    """
    all_max_diffs = []
    for path in all_paths:
        paths, _, _ = svg2paths2(path)
        single_paths = get_single_paths(paths)
        all_max_diffs.append(calc_max_diff(single_paths))
    all_max_diffs = np.array(all_max_diffs)
    total_max_diff = all_max_diffs[np.argsort(-all_max_diffs)[:index+1]][index]
    return total_max_diff

def all_paths_to_max_diffs(all_paths):
    all_max_diffs = []
    for path in all_paths:
        paths, _, _ = svg2paths2(path)
        single_paths = get_single_paths(paths)
        all_max_diffs.append(calc_max_diff(single_paths))
    return all_max_diffs

def get_viewbox(single_path, total_max_diff, offset: float = 1.0):
    abs_start = single_path.start
    abs_end = single_path.end
    top_left = complex(min(abs_start.real, abs_end.real), min(abs_start.imag, abs_end.imag))
    bottom_right = complex(max(abs_start.real, abs_end.real), max(abs_start.imag, abs_end.imag))
    diff = bottom_right - top_left
    center = top_left + diff / 2
    new_top_left = center - complex(total_max_diff / 2, total_max_diff / 2)
    viewbox = f"{new_top_left.real - offset} {new_top_left.imag - offset} {total_max_diff + offset*2} {total_max_diff + offset*2}"
    return viewbox  # "min_x min_y width height"

def get_rasterized_segments(single_paths:list, stroke_width:float, total_max_diff: float, svg_attributes, centered = False):
    if centered:
        return np.array([raster(disvg(my_path, paths2Drawing=True, stroke_widths=[stroke_width] * len(my_path), viewbox=get_viewbox(my_path, total_max_diff))) for my_path in single_paths if my_path.length() > 0.])
    else:
        viewbox=svg_attributes["viewBox"]
        return np.array([raster(disvg(my_path, paths2Drawing=True, stroke_widths=[stroke_width] * len(my_path), viewbox=viewbox)) for my_path in single_paths if my_path.length() > 0.])

def svg_path_to_segment_image_arrays(svg_path, total_max_diff: float):
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
    rasterized_segments_centered = get_rasterized_segments(single_paths, stroke_width = 0.5, total_max_diff=total_max_diff, svg_attributes=svg_attributes, centered=True)

    # everything placed where it belongs
    rasterized_segments = get_rasterized_segments(single_paths, stroke_width = 2.0, total_max_diff=total_max_diff, svg_attributes=svg_attributes, centered=False)

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
    SEGMENT_THRESHOLD = 512  # equivalent to context length
    OUT_DIR = "/scratch2/moritz_data/glyphazzn/B_simplified/numpy"
    INPUT_DIR = "/scratch2/moritz_data/glyphazzn/B_simplified/svgs"
    DESCRIPTION = ""
    OVERRIDE = True
    OUT_W = 128
    OUT_H = 128
    MODE = ["first_point_center", "whole_shape_center"]

    if OVERRIDE:
        if os.path.exists(OUT_DIR):
            input(f"you are about to delete the existing output directory {OUT_DIR}. press enter to continue")
            os.system(f"rm -rf {OUT_DIR}")

    df = pd.DataFrame(columns=["original_viewbox", "new_viewbox", "segments", 'raster_filename_absolute', "raster_filename_centered", "position_filename", 'class', 'split'])
    
    for curr_class in ["B"]:
        print(f"processing class {curr_class}")
        if not os.path.exists(OUT_DIR):
            os.makedirs(os.path.join(OUT_DIR, curr_class))


        # statistic_df = pd.read_csv("/home/mfeuerpfeil/master/thesis/datasets/figr8_trees_statistics.csv")

        # all_paths = statistic_df[statistic_df["num_segments"] < SEGMENT_THRESHOLD].file.values
        # all_paths = glob("/scratch2/moritz_data/openmoji_overfit_normalized/smile/*.svg")
        
        
        #FIXME all_paths = glob(f"{INPUT_DIR}/{curr_class}/*.svg")
        all_paths = glob(f"{INPUT_DIR}/*.svg")
        print(f"processing {len(all_paths)} paths\n")

        print("finding total max diff...")
        total_max_diff = all_paths_to_max_diff(all_paths, index=4)
        print(f"found total max diff: {total_max_diff}\n")

        for i, path in enumerate(tqdm(all_paths)):
            paths, attributes, svg_attributes = svg2paths2(path)
            single_paths = get_single_paths(paths)

            rasterized_segments_centered = get_rasterized_segments(single_paths, stroke_width = 0.5, total_max_diff=total_max_diff, svg_attributes=svg_attributes, centered=True)
            rasterized_segments = get_rasterized_segments(single_paths, stroke_width = 2.0, total_max_diff=total_max_diff, svg_attributes=svg_attributes, centered=False)

            position_information = get_positional_array_from_paths(single_paths, svg_attributes)

            assert position_information.shape[0] == rasterized_segments_centered.shape[0] == rasterized_segments.shape[0], "something went wrong"

            raster_filename_absolute = f"I{i}_{len(rasterized_segments)}_Segments_images_absolute.npy"
            raster_filename_centered = f"I{i}_{len(rasterized_segments)}_Segments_images_centered.npy"
            position_filename = f"I{i}_{len(rasterized_segments)}_Segments_positions.npy"

            np.save(os.path.join(OUT_DIR, curr_class, raster_filename_absolute), rasterized_segments)
            np.save(os.path.join(OUT_DIR, curr_class, raster_filename_centered), rasterized_segments_centered)
            np.save(os.path.join(OUT_DIR, curr_class, position_filename), position_information)
            
            new_row = {
                "path" : path,
                "original_viewbox" : svg_attributes["viewBox"],
                "new_viewbox": get_viewbox(single_paths[0], total_max_diff),
                "total_max_diff" : total_max_diff,
                "segments" : len(rasterized_segments),
                "raster_filename_absolute": raster_filename_absolute,
                "raster_filename_centered": raster_filename_centered,
                "position_filename": position_filename,
                "class": curr_class,
                "split": np.random.choice(["train", "test"], p=[0.8, 0.2])
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print("max segments:", df.segments.max())
        df.to_csv(os.path.join(OUT_DIR, 'split.csv'), index=False)