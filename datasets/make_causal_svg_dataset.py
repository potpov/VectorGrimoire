"""
This script load the FIGR-8 svg dataset https://github.com/marcdemers/FIGR-8-SVG/tree/master
it breaks each SVG image I into N sub-images where N is the total number of SVG paths for the image I.
each sub-image Ii (with i <= N) will be made of the fusion between Mi and I(i-1).
the M paths are sorted by relevance where M0 is the largest path in the image and Mn is the smallest.
"""
import random

import pandas as pd
import numpy as np
from svgpathtools import svg2paths, disvg, Path  # this is used to READ and breakdown SVG
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


# FIGR8_PATH = "/scratch4/mcipriano/SVG/FIGR-8-SVG/Data"
# FIGR8_PATH = "/scratch2/moritz_data/openmoji_black_svg"
# FIGR8_PATH = "/scratch2/moritz_data/openmoji_black_svg"
FIGR8_PATH = "/home/mfeuerpfeil/master/thesis/datasets/openmoji_overfit_svg"
OUT_DIR = "/scratch2/moritz_data/causal_openmoji_black_overfit"
OUT_W = 128
OUT_H = 128
DEBUG = False


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
    # img = np.flip(img, axis=0)  # images are rastered upside down -> wtf?
    img = np.array(img)
    img = 255 - img[:, :, 3]  # RGBA -> grey-scale
    return img.astype(np.uint8)


def update_split(group):
    """
    Generate a test split for each class using 1/4 of the whole data
    @param group: Dataframe with entries for a given class
    @return: Same dataframe with the split value updated for 25% of the samples
    """
    n_samples = int(group.shape[0] * 0.25)
    rows_to_update = group.sample(n_samples).index
    group.loc[rows_to_update, "split"] = "test"
    return group


def export_dataset(policy: str, context_length: int = 25, patience: int = 5, resume: bool = False, skip_simplify: bool = False):
    """
    For each SVG entry of the dataset create a raster versions of each path which is part of that entry.
    the policy specify how to sort and group each path. Lines which are part of a Path can also be considered
    as path themselves
    @param policy: one of: "closed", "length", "position".
    "closed" returns only closed path (sort by area), "length" sort by length of segments,
    "position" (BETA) compute the distance from the origin of the min (x, y) point in the bounding box of the path
     and use that for sorting
    @param context_length: ideal number of sub-image for each SVG path
    @param patience: threshold to discard an SVG image, images with less than this number of paths will be discarded.
    This applies only if "closed" policy is selected and after all the connected segments in the image are merged into
    paths.
    @param resume: if true, load last completed folder from the csv and resume the extraction process
    @return: 0 if export is sucessful
    """

    assert policy in ["closed", "length", "position"], "Wrong policy or policy not implemented yet!"
    print(f"Generating incremental dataset with policy: {policy}")

    folders = os.listdir(FIGR8_PATH)
    if resume:
        df = pd.read_csv(os.path.join(OUT_DIR, policy, 'split.csv'))
        start_id = len(df["class"].unique())
        print(f"Last id found in dataset {start_id} -  len of dataset: {len(folders)}")
        if start_id == len(folders):
            print("Process is already complete. Abort.")
            return 0
        assert folders[:start_id] == df["class"].unique().tolist()
        folders = folders[start_id:]  # removing processed folders
        print(f"Resuming preprocessing from folder {folders[0]}")
    else:
        start_id = 0
        df = pd.DataFrame(columns=['filename', 'class', 'split'])

    for folder_id, folder in tqdm(enumerate(folders, start=start_id), total=len(folders), initial=start_id):
        inner_bar = tqdm(
            total=len(os.listdir(os.path.join(FIGR8_PATH, folder))),
            desc=f"Processing folder for class \"{folder}\"",
            leave=False
        )
        for image_id, img_name in enumerate(os.listdir(os.path.join(FIGR8_PATH, folder))):
            file_path = os.path.join(FIGR8_PATH, folder, img_name)
            # if img_name.replace(".svg", "") not in ['1F642']:
            #     continue
            paths, attributes = svg2paths(file_path)

            if len(paths) == 0:  # empty SVG
                continue

            if DEBUG:
                plt.imshow(raster(disvg(paths, paths2Drawing=True)))
                plt.show()

            if not skip_simplify:
                # using path as they are results in a loss of very important features
                # taking all the lines as paths and then grouping connected lines is the best option
                # also: a closed path must be formed by 1 or more connected lines anyway!
                paths = [item for sublist in paths for item in sublist]  # flatten paths
                paths = Path(*paths).continuous_subpaths()  # grouping connected lines

            if policy == "closed":
                paths = [p for p in paths if p.isclosed()]
                if len(paths) < patience:
                    continue
                sort_attrib = [np.abs(p.area()) for p in paths]  # taking paths with larger areas first
            elif policy == "length":
                sort_attrib = [Path(p).length() for p in paths]
            elif policy == "position":
                sort_attrib = []
                for p in paths:
                    bbox = p.bbox()  # returns x_min, x_max, y_min, y_max
                    sort_attrib.append(np.sqrt(bbox[0] ** 2 + bbox[2] ** 2))
            else:
                raise Exception("Wrong policy or policy not implemented yet!")

            # sorting and truncating regardless the policy used
            # paths = [p for _, p in sorted(zip(sort_attrib, paths), reverse=True)]
            # using index of enumerate as second sorting params if two occurrences are the same
            paths = [p for _, p in sorted(enumerate(paths), key=lambda x: (sort_attrib[x[0]], x[0]), reverse=True)]
            attributes = [attributes[i] for i, p in sorted(enumerate(paths), key=lambda x: (sort_attrib[x[0]], x[0]), reverse=True)]
            paths = paths[:context_length]  # do not exceed CL
            attributes = attributes[:context_length]  # do not exceed CL

            # this caused issues where the default fill was true
            for i, attr in enumerate(attributes):
                # this solves a weird issue that if just "d" is part of the attributes, the defaults are desired
                if len(attr.keys()) == 1:
                    print("only: ",attr.keys())
                    continue
                # if more than just "d" is part of the attributes, we have to make sure the defaults are disabled
                if "fill" not in attr:
                    attributes[i]["fill"] = "none"
                if "stroke" not in attr:
                    attributes[i]["stroke"] = "#000"
            if skip_simplify:
                # merging all the paths but hiding each of them
                svg = disvg(paths, paths2Drawing=True, attributes = attributes)  # merging all the paths
            else:
                # merging all the paths but hiding each of them
                svg = disvg(paths, paths2Drawing=True)  # merging all the paths
            for i in range(1, len(paths) + 1):  # element 0 is Def
                svg.elements[i].attribs["visibility"] = "hidden"

            # Saving each path -> using visibility attribs in the
            imgs = []
            for i in range(1, len(paths) + 1):
                del svg.elements[i].attribs["visibility"]  # showing this path
                img = raster(svg)
                imgs.append(img)
                svg.elements[i].attribs["visibility"] = "hidden"  # restore
                if DEBUG:
                    plt.imshow(img)
                    plt.show()
            filename = f"F{folder_id}_I{image_id}_P{len(imgs)}.npy"
            np.save(os.path.join(OUT_DIR, policy, filename), np.stack(imgs))
            new_row = {
                "filename": filename,
                "class": folder,
                "split": "train"
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            inner_bar.update(1)

        df.to_csv(os.path.join(OUT_DIR, policy, 'split.csv'), index=False)  # backup at the end of every folder
        inner_bar.close()

    # update test samples and save final splits
    df = df.groupby("class").apply(update_split)
    df.to_csv(os.path.join(OUT_DIR, policy, 'split.csv'), index=False)
    return 0


def compute_stats():
    """
    show the distribution of the number of path and the number of lines foreach path in the dataset.
    @return: None
    """
    path_count = []
    line_count = []
    sample_num = 5000
    max_value = 60
    folders = list(os.listdir(FIGR8_PATH))
    random.shuffle(folders)
    folders = folders[:sample_num]
    for folder in tqdm(folders, total=len(folders)):
        img_name = random.choice(os.listdir(os.path.join(FIGR8_PATH, folder)))
        paths, attributes = svg2paths(os.path.join(FIGR8_PATH, folder, img_name))
        path_count.append(min(max_value, len(paths)))
        for path in paths:
            line_count.append(min(max_value, len(path)))

    for target, value in {"Path": path_count, "Line": line_count}.items():
        plt.hist(value, bins=60, edgecolor='black')
        plt.title(f'{target} distribution ({sample_num} random samples)')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()

def compute_openmoji_stats():
    """
    show the distribution of the number of path and the number of lines foreach path in the dataset.
    @return: None
    """
    path_count = []
    line_count = []
    sample_num = 4082
    max_value = 60
    folders = list(os.listdir(FIGR8_PATH))
    random.shuffle(folders)
    # folders = folders[:sample_num]
    for folder in tqdm(folders, total=len(folders)):
        dir_list = os.listdir(os.path.join(FIGR8_PATH, folder))
        for i in range(sample_num):
            img_name = dir_list[i]
            paths, attributes = svg2paths(os.path.join(FIGR8_PATH, folder, img_name))
            path_count.append(min(max_value, len(paths)))
            for path in paths:
                line_count.append(min(max_value, len(path)))

    for target, value in {"Path": path_count, "Line": line_count}.items():
        plt.hist(value, bins=60, edgecolor='black')
        plt.title(f'{target} distribution ({sample_num} random samples)')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Export for SVG dataset')
    parser.add_argument('--policy', '-p', help='policy for grouping', default='length')
    parser.add_argument('--context_len', '-l', help='max sub-images', default=25)
    parser.add_argument('--resume', '-r', action='store_true', help='resume export (default false)')
    args = parser.parse_args()

    # num_svg * num_image_per_svg (UB 10) * H * W * 8 bit -> convert to gigabyte
    num_files = sum([len(os.listdir(os.path.join(FIGR8_PATH, p))) for p in os.listdir(FIGR8_PATH)])
    # num_files = len(glob(FIGR8_PATH, recursive=True))
    print(f"The number of unique images in this dataset is: {num_files}.")
    print(f"Maximum output size with current configuration is: "
          f"{round(num_files * int(args.context_len) * OUT_H * OUT_W  * 8 * 1.25e-10, 2)} gigabyte")

    if args.policy == "position":
        print("Warning. position policy is still in beta.")
    LinuxPath(os.path.join(OUT_DIR, args.policy)).mkdir(parents=True, exist_ok=True)
    export_dataset(policy=args.policy, context_length=int(args.context_len), resume=args.resume, skip_simplify=True)




