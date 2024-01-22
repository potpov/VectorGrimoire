from typing import List
from torchvision import transforms
from PIL import Image
import wandb
import numpy as np
import torch
from torch import Tensor
import os
from torchvision.utils import make_grid
from torchvision.transforms import Resize
from svgwrite import Drawing
from svgpathtools import disvg, CubicBezier

def calculate_global_positions(local_positions: Tensor, local_viewbox_width:float, global_center_positions: Tensor):
    """
    Calculates the global positions of svg shapes from the local centered positions.
    """
    local_points_delta_to_middle = local_positions - 0.5
    scaled_local_points_delta_to_middle = local_points_delta_to_middle * local_viewbox_width
    global_center_positions = global_center_positions.unsqueeze(1).unsqueeze(1).repeat(1, scaled_local_points_delta_to_middle.shape[1], scaled_local_points_delta_to_middle.shape[2], 1)
    global_positions = global_center_positions + scaled_local_points_delta_to_middle
    return global_positions

def tensor_to_complex(my_tensor):
    return complex(my_tensor[0].item(), my_tensor[1].item())

def stroke_points_to_bezier(my_tensor:Tensor):
    """
    expects my_tensor to be in shape (4, 2)
    """
    return CubicBezier(tensor_to_complex(my_tensor[0]), tensor_to_complex(my_tensor[1]), tensor_to_complex(my_tensor[2]), tensor_to_complex(my_tensor[3]))

def shapes_to_drawing(shapes:Tensor, stroke_width:float, w=128) -> Drawing:
    """
    expects shapes to be in shape (n, 4, 2)
    """
    all_shapes = []
    for shape in shapes:
        all_shapes.append(stroke_points_to_bezier(shape))
    drawing = disvg(all_shapes, stroke_widths=[stroke_width]*len(all_shapes), paths2Drawing=True, viewbox=f"0 0 {w} {w}")
    return drawing

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer.buffer_rgba())
    return X[:,:,:3]


def make_tensor(x, grad=False):
    x = torch.tensor(x, dtype=torch.float32)
    x.requires_grad = grad
    return x

def log_all_images(images: List[Tensor], log_key="validation", caption="Captions not set"):
    """
    Logs all images of a list as grids to wandb.

    Args:
        - images (List[Tensor]): List of images to log
        - log_key (str): key for wandb logging
        - captions (str): caption for the images
    """
    if get_rank() != 0:
        return

    assert len(images) > 0, "No images to log"

    common_size = images[0].shape[-2:]
    resizer = Resize(common_size, antialias=True)

    image_result = make_grid(images[0], nrow=4, padding=5, pad_value=0.2)
    for image in images[1:]:
        image_result = torch.concat((image_result, make_grid(resizer(image), nrow=4, padding=5, pad_value=0.2)), dim=-1)

    wandb.log({log_key: wandb.Image(image_result, caption=caption)})

def log_images(recons: Tensor, real_imgs: Tensor, log_key="validation", captions="Captions not set"):

    # if get_rank() != 0:
    #     return

    if recons.shape[-2:] != real_imgs.shape[-2:]:
        common_size = recons.shape[-2:]
        resizer = Resize(common_size, antialias=True)
        real_imgs_resized = resizer(real_imgs)
    else:
        real_imgs_resized = real_imgs

    bs, c, w, h = real_imgs_resized.shape

    if recons.shape[1] > real_imgs_resized.shape[1]:
        real_imgs_resized = torch.cat((real_imgs_resized, torch.ones((bs, 1, w, h), device=real_imgs_resized.device)), dim=1)
    elif recons.shape[1] < real_imgs_resized.shape[1]:
        recons = torch.cat((recons, torch.ones((bs, 1, w, h), device=recons.device)), dim=1)

    image_result = torch.concat((
        make_grid(real_imgs_resized, nrow=4, padding=5, pad_value=0.2),
        make_grid(recons, nrow=4, padding=5, pad_value=0.2)
        ),
        dim=-1
    )

    wandb.log({log_key: wandb.Image(image_result, caption=captions)})


def get_rank() -> int:
    if not torch.distributed.is_available():
        return 0  # Training on CPU
    if not torch.distributed.is_initialized():
        rank = os.environ.get("LOCAL_RANK")  # from pytorch-lightning
        if rank is not None:
            return int(rank)
        else:
            return 0
    else:
        return torch.distributed.get_rank()

def tensor_to_histogram_image(tensor, bins=100):
    # Create a histogram plot
    plt.hist(tensor, bins=bins)
    plt.title('Codebook usage histogram')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Create a PIL image from the BytesIO object
    image = Image.open(buf).copy()

    # Close the buffer
    buf.close()

    return image

##############################################################################################################
# SVG splitting utils
##############################################################################################################
from svgpathtools import svg2paths, svg2paths2, disvg, Path  # this is used to READ and breakdown SVG
import math
from svgwrite import Drawing
from cairosvg import svg2png
import io
from matplotlib import pyplot as plt
import copy
from torchvision import transforms
def raster(svg_file: Drawing, out_h: int = 128, out_w: int = 128):
    """
    This function simply resizes and rasters a series of Paths
    @param svg_file: Drawing object
    @return: Numpy array of the raster image single-channel
    """
    svg_png_image = svg2png(
        bytestring=svg_file.tostring(),
        output_width=out_w,
        output_height=out_h,
        background_color="white")
    img = Image.open(io.BytesIO(svg_png_image))
    # rgb_image = Image.new("RGB", img.size, (255, 255, 255))
    # rgb_image.paste(img, mask=img.split()[3])
    transform = transforms.ToTensor()
    tensor_image = transform(img)
    return tensor_image

def save_path_as_image(path: Path, out_h: int = 128, out_w: int = 128):
    """
    This function simply resizes and rasters a series of Paths
    @param svg_file: Drawing object
    @return: Numpy array of the raster image single-channel
    """
    svg_file = disvg(path, paths2Drawing=True, stroke_widths=[2.0] * len(path))
    svg_png_image = svg2png(
        bytestring=svg_file.tostring(),
        output_width=out_w,
        output_height=out_h,
        background_color="white")
    img = Image.open(io.BytesIO(svg_png_image))
    img.save("test.png")

def plot_segments(rasterized_segments, title:str="A disassembled tree"):
    assert rasterized_segments.shape[0] > 8, "too few segments to plot"
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
    """
    returns viewbox and center of the viewbox as x-y-tuple
    """
    abs_start = single_path.start
    abs_end = single_path.end
    top_left = complex(min(abs_start.real, abs_end.real), min(abs_start.imag, abs_end.imag))
    bottom_right = complex(max(abs_start.real, abs_end.real), max(abs_start.imag, abs_end.imag))
    diff = bottom_right - top_left
    center = top_left + diff / 2
    new_top_left = center - complex(total_max_diff / 2, total_max_diff / 2)
    viewbox = f"{new_top_left.real - offset} {new_top_left.imag - offset} {total_max_diff + offset*2} {total_max_diff + offset*2}"
    return viewbox, [center.real, center.imag]

def get_rasterized_segments(single_paths:list, stroke_width:float, total_max_diff: float, svg_attributes, centered = False, height: int = 128, width: int = 128) -> List:
    if centered:
        out = [get_viewbox(my_path, total_max_diff) for my_path in single_paths if my_path.length() > 0.]
        viewboxes = [x[0] for x in out]
        centers = [x[1] for x in out]
        rasterized_segments = [raster(disvg(my_path, paths2Drawing=True, stroke_widths=[stroke_width] * len(my_path), viewbox=viewboxes[i]), out_h = height, out_w = width) for i, my_path in enumerate(single_paths) if my_path.length() > 0.]
        return rasterized_segments, centers
    else:
        viewbox=svg_attributes["viewBox"]
        rasterized_segments = [raster(disvg(my_path, paths2Drawing=True, stroke_widths=[stroke_width] * len(my_path), viewbox=viewbox), out_h = height, out_w = width) for my_path in single_paths if my_path.length() > 0.]
        centers = [(0,0)] * len(rasterized_segments)
        return rasterized_segments, centers


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

def get_similar_length_paths(queue:list, max_length:float):
    similar_length_paths = []
    curr_aggregated_path = Path()
    while len(queue) > 0:
        path = queue.pop(0)
        if curr_aggregated_path.length() + path.length() < max_length and curr_aggregated_path.end == path.start:
            curr_aggregated_path.extend(path)
        else:
            similar_length_paths.append(curr_aggregated_path)
            curr_aggregated_path = path
    return similar_length_paths[1:]  # first path is always empty

def check_for_continouity(single_paths: list):
    for path in single_paths:
        if not path.iscontinuous():
            return False
    return True