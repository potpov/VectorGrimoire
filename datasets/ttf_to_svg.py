#!python3
# Convert character of TTF to SVG.
# Author: "Gary Lee" <garywlee@gmail.com>
# Adapted by: Moritz F.

# Usage: ttf2svg.py [char] [OUTPUT_FILE] [TTF_FONT]
# Example: ttf2svg.py A A.svg /Library/Fonts/arial.ttf'
#
# Requirement:
# - pip3 install freetype-py
# - pip3 install svgpathtools
# - files downloaded & unzipped from https://github.com/google/fonts/archive/main.zip
#
# References:
# - https://www.freetype.org/freetype2/docs/reference/ft2-outline_processing.html#ft_outline_decompose
# - https://gist.github.com/p3t3r67x0/a35e9e0e9f6f22053e8f7a5543b59724

import sys
from freetype import Face, FT_Curve_Tag, FT_Curve_Tag_On, FT_Vector
from svgpathtools import (wsvg, Line, CubicBezier, QuadraticBezier, Path)
from glob import glob
import re
from tqdm import tqdm
import os
from numpy.random import choice
import pandas as pd
import yaml
import ast


class TtfSvgConverter:
    VERBOSE = False
    STROKE_WIDTHS = 10
    CHAR_WIDTH = 48
    CHAR_HEIGHT = 64
    CHAR_SIZE = CHAR_WIDTH * CHAR_HEIGHT
    def __init__(self, ttfPath=None):
        self.ttfPath = ttfPath
        self.reset()

    def reset(self):
        self.svgPath = []
        self._lastX = 0
        self._lastY = 0

    def _verbose(self, *args):
        if self.VERBOSE:
            print(*args)

    def lastXyToComplex(self):
        return self.tupleToComplex((self._lastX, self._lastY))

    def tupleToComplex(self, xy):
        return xy[0] + xy[1] * 1j

    def vectorToComplex(self, v):
        return v.x + v.y * 1j

    def vectorsToPoints(self, vectors):
        return [(v.x, v.y) for v in vectors if v is not None]

    def callbackMoveTo(self, *args):
        self._verbose('MoveTo ', len(args), self.vectorsToPoints(args))
        self._lastX, self._lastY = args[0].x, args[0].y

    def callbackLineTo(self, *args):
        self._verbose('LineTo ', len(args), self.vectorsToPoints(args))
        line = Line(self.lastXyToComplex(), self.vectorToComplex(args[0]))
        self.svgPath.append(line)
        self._lastX, self._lastY = args[0].x, args[0].y

    def callbackConicTo(self, *args):
        self._verbose('ConicTo', len(args), self.vectorsToPoints(args))
        curve = QuadraticBezier(self.lastXyToComplex(), self.vectorToComplex(args[0]), self.vectorToComplex(args[1]))
        self.svgPath.append(curve)
        self._lastX, self._lastY = args[1].x, args[1].y

    def callbackCubicTo(self, *args):
        self._verbose('CubicTo', len(args), self.vectorsToPoints(args))
        curve = CubicBezier(self.lastXyToComplex(), self.vectorToComplex(args[0]), self.vectorToComplex(args[1]), self.vectorToComplex(args[2]))
        self.svgPath.append(curve)
        self._lastX, self._lastY = args[2].x, args[2].y

    def calcViewBox(self, path):
        xmin, xmax, ymin, ymax = path.bbox()
        xmin, xmax, ymin, ymax = xmin - self.CHAR_WIDTH, xmax + self.CHAR_WIDTH, ymin - self.CHAR_HEIGHT, ymax + self.CHAR_HEIGHT
        dx = xmax - xmin
        dy = ymax - ymin
        viewbox = '{} {} {} {}'.format(xmin, ymin, dx, dy)
        return viewbox

    def generate(self, text, output):
        self.reset()
        face = Face(self.ttfPath)
        face.set_char_size(self.CHAR_SIZE)
        for ch in text:
            face.load_char(ch)
            outline = face.glyph.outline
            outline.decompose(context=None, move_to=self.callbackMoveTo, line_to=self.callbackLineTo, conic_to=self.callbackConicTo, cubic_to=self.callbackCubicTo)
            path = Path(*self.svgPath).scaled(1, -1)
            viewbox = self.calcViewBox(path)
            attr = {
                'width': '100%',
                'height': '100%',
                'viewBox': viewbox,
                'preserveAspectRatio': 'xMidYMid meet'
            }
            wsvg(paths=path, colors=['#000000'], svg_attributes=attr, stroke_widths=[self.STROKE_WIDTHS], filename=output)
            break # Only handle the first character.


if __name__ == "__main__":
    # can be generated with import string; string.printable[:62]
    ALL_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    with open("font_paths.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    for dataset, params in config["fonts"].items():

        all_files = glob(params["fonts_dir"] + "/**/*.ttf", recursive=True)
        print(f"converting {dataset} to SVG, processing {len(all_files)} files")

        columns = ["char", "font_path", "font_name", "output_string", "split", "output_path", "conversion_skipped"]
        df = pd.DataFrame(columns=columns)

        csv_files = [file for file in os.listdir(params["svg_dir"]) if file.startswith('split_') and file.endswith('.csv')]

        start_from = 0
        total_iterations_df = 0
        metadata = None
        if os.path.exists(os.path.join(params["fonts_dir"], "metadata.csv")):
            metadata = pd.read_csv(os.path.join(params["fonts_dir"], "metadata.csv"))

        total_skip = 0

        for i, font_path in tqdm(enumerate(all_files), total=len(all_files)):
            if i < start_from:
                continue
            converter = TtfSvgConverter(ttfPath=font_path)
            tags = []
            for char in ALL_CHARS:
                conversion_skipped = "no"
                font_name = font_path.split("/")[-1].split(".")[0]
                if dataset == "fonts-main":
                    output_string = re.sub(r'\[.*?\]', '', font_name)
                elif dataset == "glyphazzn":
                    output_string = font_name
                elif dataset in ["allfonts", "dafont"]:
                    output_string = font_name.strip().replace(" ", "_").lower()
                    tags = []
                    if metadata:
                        try:
                            parent_dir = os.path.basename(os.path.dirname(font_path))
                            tags_str = metadata.loc[metadata['filename'] == parent_dir, 'tags'].iloc[0]  # try to get classes for this font
                            tags = ast.literal_eval(tags_str)  # don't really need that here
                        except Exception as e:
                            pass
                else:
                    raise Exception(f"unknown dataset {dataset}")

                split = choice(["train", "test"], p=[0.8, 0.2])
                output_folder = os.path.join(params["svg_dir"], split, char)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output = os.path.join(output_folder, f"{char}_{output_string}.svg")
                # the splitting is not seeded, so we need to check if the file already exists in any split
                if os.path.exists(output) or os.path.exists(output.replace("test", "train")) or os.path.exists(output.replace("test", "train")):
                    continue
                try:
                    converter.generate(char, output)
                except Exception as e:
                    conversion_skipped = "yes"
                    if isinstance(e, KeyboardInterrupt):
                        break
                    else:
                        total_skip += 1
                        print(f"skipped {font_path}")

                # add to dataframe
                new_row = pd.DataFrame({
                    "char": [char],
                    "font_path": [font_path],
                    "font_name": [font_name],
                    "output_string": [output_string],
                    "split": [split],
                    "output_path": [output],
                    "conversion_skipped": [conversion_skipped],
                    "tags": [tags]
                })

                df = pd.concat([df, new_row], ignore_index=True)

            if i % 2000 == 0:
                out_df_path = os.path.join(params["svg_dir"], f"split_{total_iterations_df}th_iteration.csv")
                while os.path.exists(out_df_path):
                    total_iterations_df = total_iterations_df + 1
                    out_df_path = os.path.join(params["svg_dir"], f"split_{total_iterations_df}th_iteration.csv")
                df.to_csv(out_df_path, index=False)
                total_iterations_df += 1
                df = pd.DataFrame(columns=columns)
            else:
                df.to_csv(os.path.join(params["svg_dir"], f"split_{total_iterations_df}th_iteration.csv"), index=False)
        print(f"total skipped: {total_skip}")