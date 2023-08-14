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


FIGR8_PATH = "/scratch4/mcipriano/SVG/FIGR-8-SVG/Data"
OUT_W = 500
OUT_H = 500


def raster(svg_file):
    svg_png_image = svg2png(
        bytestring=svg_file.tostring(),
        output_width=OUT_W,
        output_height=OUT_H)
    img = Image.open(io.BytesIO(svg_png_image))
    img = np.flip(img, axis=0)  # images are rastered upside down -> wtf?
    return img


def export_dataset():
    df = []
    id_counter = 0
    for folder in os.listdir(FIGR8_PATH):
        for img_name in os.listdir(os.path.join(FIGR8_PATH, folder)):
            file_path = os.path.join(FIGR8_PATH, folder, img_name)
            paths, attributes = svg2paths(file_path)

            # Using groups of lines if we have only one path
            if len(paths) == 1:
                if len(paths[0]) < 10:
                    print(f"skip {file_path}, too few lines")
                    continue
                n_take = round(len(paths[0]) / 5)  # check how many lines to use to have a total of 5 paths
                paths = [Path(*paths[0][i * n_take: (i + 1) * n_take]) for i in range(5)]  # break lines into paths
                # paths = [Path(paths[0][i * n_take: (i + 1) * n_take]) for i in range(5)]
            # TODO: sorting of the paths per importance

            # merging all the paths but hiding each of them
            svg = disvg(paths, paths2Drawing=True)  # merging all the paths
            for i in range(1, len(paths) + 1):  # element 0 is Def
                svg.elements[i].attribs["visibility"] = "hidden"

            # Saving each path -> using visibility attribs in the
            for i in range(1, len(paths) + 1):
                del svg.elements[i].attribs["visibility"]  # showing this path
                img = raster(svg)
                svg.elements[i].attribs["visibility"] = "hidden"  # restore
                plt.imshow(img)
                plt.show()


def compute_stats():
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


if __name__ == '__main__':
    export_dataset()




