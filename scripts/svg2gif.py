import torch
from svgpathtools import svg2paths
from tqdm import tqdm
from utils import get_rasterized_segments, svg2paths2, get_single_paths
import imageio
from pathlib import Path
from cairosvg import svg2png
from PIL import Image
import io
import numpy as np
import os
import natsort
from utils import raster
from svgpathtools import disvg


def svg_to_gif(svg_file, gif_file, context=0):
    # Read paths from SVG
    paths, attributes, svg_attributes = svg2paths2(svg_file)
    W, H = 250, 250
    global_colours = [(0, 0, 0)] * len(paths)

    imgs, _ = get_rasterized_segments(
        paths[context:],
        stroke_width=1,
        total_max_diff=10,
        svg_attributes=svg_attributes,
        centered=False,
        height=H,
        width=W
    )

    if context != 0:
        context_len = len(paths[:context])
        pred_len = len(paths) - context_len
        context_img = raster(
            disvg(
                paths[:context],
                paths2Drawing=True,
                stroke_widths=[1] * context_len,
                viewbox=svg_attributes["viewBox"],
                colors=[(255, 0, 0)] * context_len
            ),
            out_h=H,
            out_w=W
        )
        imgs = [context_img] + imgs
        global_colours = [(255, 0, 0)] * context_len + [(0, 0, 0)] * pred_len

    ## if we have context, global colours will differ from pred
    final = raster(
        disvg(
            paths,
            paths2Drawing=True,
            stroke_widths=[1] * len(paths),
            viewbox=svg_attributes["viewBox"],
            colors=global_colours
        ),
        out_h=H,
        out_w=W
    )
    final = (final.moveaxis(0, -1).numpy() * 255).astype(np.uint8)

    # move to WH3 and converting to numpy
    imgs = [1 - (i.moveaxis(0, -1).numpy()) for i in imgs]

    accumulated_sum = np.zeros_like(imgs[0])
    for i in range(len(imgs)):
        imgs[i] = imgs[i] + accumulated_sum
        imgs[i][imgs[i] > 1] = 1  # thresholding
        accumulated_sum = accumulated_sum + imgs[i]

    # moving into 0-255 range and inverting white and black again
    imgs = [((1 - i) * 255).astype(np.uint8) for i in imgs]
    imgs = imgs + [final]
    imageio.mimsave(gif_file, imgs, loop=1, duration=3 / len(imgs))

for context in [0]:
# for context in [6, 12]:
    main_dir = f"/Users/marcocipriano/Desktop/ECCV SVG/benchmark/stage2/figr8/temp_0_1/vq_context_{context}_t0"
    # main_dir = f"/Users/marcocipriano/Desktop/ECCV SVG/benchmark/stage2/figr8/temp_0/vq_context_{context}_t0"

    with open(os.path.join(main_dir, "svgs", "prompts.txt")) as f:
        prompts = f.readlines()

    in_subdir = os.path.join(main_dir, "svgs", "pi_fixed")
    out_subdir = os.path.join(main_dir, "gifs", "pi_fixed")
    Path(out_subdir).mkdir(parents=True, exist_ok=True)
    files = os.listdir(in_subdir)
    files = [f for f in files if f != ".DS_Store"]
    files = natsort.natsorted(files)  # sorting like finder
    for p_id, filename in tqdm(enumerate(files), total=len(files)):
        svg_file = os.path.join(in_subdir, filename)
        gif_path = os.path.join(out_subdir, f"{p_id}_{prompts[p_id].strip()}.gif")
        svg_to_gif(
            svg_file=svg_file,
            gif_file=gif_path,
            context=context
        )
