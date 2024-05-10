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
from svgpathtools import disvg, CubicBezier, Line
import cairosvg
from PIL import Image
from io import BytesIO
from torchvision.transforms import ToTensor
import re
import matplotlib.colors as mcolors

def get_color_gradient(num_colors: int, start_color="red", end_color="blue"):
    gradient = mcolors.LinearSegmentedColormap.from_list('gradient', [start_color, end_color])
    colors = [gradient(i / num_colors) for i in range(num_colors)]
    hex_colors = [mcolors.rgb2hex(color) for color in colors]
    return hex_colors

def get_rendered_svg_with_gradient(svg_path):
    base_attribute = {
        "fill": "none",
        "fill-opacity": "1.0",
        "filling": "0",
        "stroke":"black",
        "stroke-width":"1",
    }
    indicator_attribute = {
        "fill": "none",
        "fill-opacity": "1.0",
        "filling": "0",
        "stroke":"black",
        "stroke-width":"0.5",
        "stroke-opacity" : "0.5",
    }
    paths, attributes, svg_attributes = svg2paths2(svg_path)
    indicators = [Line(x[0].start,end=complex(0.,0.)) for x in paths]
    flattened_paths = get_flattened_paths(paths)

    num_paths = len(flattened_paths)
    gradient = get_color_gradient(num_paths, start_color = "red",end_color="black")
    new_attributes=[]
    for i in range(num_paths):
        # Calculate the color for the current path based on the gradient
        color = gradient[i]

        # Create a separate attribute dictionary for the current path
        path_attribute = base_attribute.copy()
        path_attribute['stroke'] = color

        # Add the attribute dictionary to the list
        new_attributes.append(path_attribute)

    img = drawing_to_tensor(disvg(flattened_paths + indicators, attributes=new_attributes + [indicator_attribute]*len(indicators), paths2Drawing=True))
    # Use the attributes list when calling disvg
    return img

def svg_file_path_to_tensor(path, permuted = False, plot=False, stroke_width=0.5, filling:bool=False,image_size:int=224):
    paths, attributes, svg_attributes = svg2paths2(path)
    for i, attr in enumerate(attributes):
        attr["stroke_width"] = f"{stroke_width}"
        attr["fill"] = "black" if filling else "none"

    if "viewbox" in svg_attributes:
        viewbox = svg_attributes["viewbox"]
    else:
        viewbox = None
    return_tensor = raster(disvg(paths, attributes=attributes,paths2Drawing=True, viewbox=viewbox), out_h=image_size, out_w = image_size)

    if permuted:
        return_tensor = return_tensor.permute(1,2,0)
    if plot:
        plt.imshow(return_tensor)
    return return_tensor

def add_points_to_image(all_points:Tensor, image:Tensor, image_scale:int):
    """
    inputs:
    - all_points: tensor of shape (batch, n_points, 2)
    - image: tensor of shape (batch, 3, 128, 128)
    - image_scale: 128 if the image is 128x128, 224 if the image is 224x224

    this function should be used to add predicted points to a reconstructed image for better debugging of shape predictions. 
    start/end points are red, bending control points are green.
    """
    all_points = all_points.detach().clone()
    image = image.detach().clone()
    for batch in range(all_points.shape[0]):
        for i, point in enumerate(all_points[batch][0]):
            point = point * image_scale
            point = point.long()
            # this could crash if the point is outside the image or on the border
            # try:
            radius = 2
            if i%3 == 0:
                image[batch, 0, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius] = 1
                image[batch, 1, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius] = 0
                image[batch, 2, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius] = 0
            elif i<3:
                image[batch, 0, point[1]-radius:point[1]+1, point[0]-1:point[0]+1] = 0
                image[batch, 1, point[1]-radius:point[1]+1, point[0]-1:point[0]+1] = 1
                image[batch, 2, point[1]-radius:point[1]+1, point[0]-1:point[0]+1] = 0
            elif i>3:
                image[batch, 0, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+1] = 0
                image[batch, 1, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+1] = 0.5
                image[batch, 2, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+1] = 1

            # except Exception as e:
            #     print("[INFO] couldnt add points to logging image", e)
    return image

def svg_string_to_tensor(svg_string):
    # Convert SVG string to PNG bytes
    png_bytes = cairosvg.svg2png(bytestring=svg_string, background_color="white")
    
    # Convert PNG bytes to PIL Image
    image = Image.open(BytesIO(png_bytes))
    
    # Ensure the image is in RGB mode
    image = image.convert("RGB")
    
    # Convert the PIL Image to a PyTorch tensor with three channels
    tensor = ToTensor()(image)
    
    return tensor

def get_side_by_side_reconstruction(model, dataset, idx, device, w=480):
    """
    model must be Vector_VQVAE
    dataset must be GlyphazznStage1Dataset
    """
    # Get the ground truth SVG drawing
    gt = dataset._get_full_svg_drawing(idx, width=w, as_tensor=True)

    # Reconstruct the SVG drawing
    patches, labels, positions, _ = dataset._get_full_item(idx)
    patches = patches.to(device)
    positions = positions.to(device)
    _, recons_rastered_drawing = model.reconstruct(patches, positions, dataset.individual_max_length +2, dataset.stroke_width, rendered_w=w)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Plot reconstructed drawing
    axes[1].imshow(recons_rastered_drawing.permute(1, 2, 0))
    axes[1].set_title('Reconstructed SVG')

    # Plot ground truth drawing
    axes[0].imshow(gt.permute(1, 2, 0))
    axes[0].set_title('Ground Truth SVG')
    for ax in axes:
        ax.axis('off')
     
    fig.tight_layout()
    img = fig2img(fig)
    plt.close(fig)

    return img

def drawing_to_tensor(drawing: Drawing):
    return svg_string_to_tensor(drawing.tostring())

def svg_to_tensor(file_path, new_stroke_width:float = None):
    if new_stroke_width is None:
        png_data = cairosvg.svg2png(url=file_path, background_color="white")
        image = Image.open(BytesIO(png_data))
        image = image.convert("RGB")
        tensor = ToTensor()(image)
    else:
        with open(file_path, "r") as file:
            svg_string = file.read()
        pattern = r'stroke-width="[^"]*"'
        replacement_string  = f'stroke-width="{new_stroke_width}"'
        new_svg_content = re.sub(pattern, replacement_string, svg_string)
        tensor = svg_string_to_tensor(new_svg_content)
    return tensor

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

def stroke_to_path(my_tensor: Tensor):
    """
    expects my_tensor to be in shape (1+3*num_segments, 2)
    """
    num_segments = (my_tensor.shape[0] - 1) // 3
    all_paths = []
    for seg_idx in range(num_segments):
        start_idx = seg_idx * 3
        end_idx = (seg_idx+1) * 3 + 1
        all_paths.append(stroke_points_to_bezier(my_tensor[start_idx:end_idx]))
    return Path(*all_paths)

def shapes_to_drawing(shapes:Tensor, stroke_width:float|List, w=128, num_strokes_to_paint:int = 0, linecap="round", linejoin="round") -> Drawing:
    """
    expects shapes to be in shape (n, 1+3*num_segments, 2)
    """
    assert linecap in ["round", "butt", "square"], "linecap must be either 'round', 'butt' or 'square'."
    assert linejoin in ["round", "bevel", "miter"], "linejoin must be either 'round', 'bevel' or 'miter'."

    base_attribute = {
        "fill": "none",
        "fill-opacity": "1.0",
        "filling": "0",
        "stroke":"black",
        "stroke-width":"1",
        "stroke-linecap":linecap,
        "stroke-linejoin" : linejoin

    }
    if shapes.mean() < 2.0:
        shapes = shapes * 72
    assert shapes.mean() > 1.0 and shapes.mean() < 72.0, "shapes should be already scaled in range 0. - 72."
    all_shapes = []
    for shape in shapes:
        all_shapes.append(stroke_to_path(shape))
    if num_strokes_to_paint > len(all_shapes):
        num_strokes_to_paint = len(all_shapes)
    colors = ["red"] * num_strokes_to_paint + ["black"] * (len(all_shapes) - num_strokes_to_paint)
    if isinstance(stroke_width, float):
        stroke_widths = [stroke_width] * len(all_shapes)
    elif isinstance(stroke_width, list):
        stroke_widths = stroke_width
    all_attributes = []
    for i, shape in enumerate(all_shapes):
        attributes = base_attribute.copy()
        attributes["stroke-width"] = f"{stroke_widths[i]}"
        attributes["stroke"] = colors[i]
        all_attributes.append(attributes)
    drawing = disvg(all_shapes, attributes=all_attributes, paths2Drawing=True, viewbox=f"0 0 72 72", dimensions=(w, w))  # I think the 72 comes from the simplified svg files
    return drawing

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

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

    assert len(images) > 0, "No images to log"

    common_size = images[0].shape[-2:]
    resizer = Resize(common_size, antialias=True)

    image_result = make_grid(images[0], nrow=4, padding=5, pad_value=0.2)
    for image in images[1:]:
        image_result = torch.concat((image_result, make_grid(resizer(image), nrow=4, padding=5, pad_value=0.2)), dim=-1)

    return log_key, image_result
    # return log_key, wandb.Image(image_result, caption=caption)
    # wandb.log({log_key: wandb.Image(image_result, caption=caption)})

def get_merged_image_for_logging(images: List[Tensor]) -> Tensor:
    """
    resized and merges all images of a list into a single loggable tensor
    """
    common_size = images[0].shape[-2:]
    resizer = Resize(common_size, antialias=True)
    images = [resizer(image) for image in images]

    merged_image = make_grid(images, nrow=math.ceil(np.sqrt(len(images))), padding=5, pad_value=0.2)

    return merged_image


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
    # return log_key, wandb.Image(image_result, caption=captions)
    # WandbLogger.log_image(key=log_key, images=image_result, caption=captions)
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
    single_paths = [Path(segment) for path in paths for segment in path._segments]
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

def get_rasterized_segments(single_paths:list, stroke_width:float, total_max_diff: float, svg_attributes, centered = False, height: int = 128, width: int = 128, colors=None) -> List:
    if centered:
        single_paths = [my_path for my_path in single_paths if my_path.length() > 0.]
        if len(single_paths) == 0:
            # print("[INFO] tried to rasterize an empty path")
            return [torch.ones((3, height, width)), torch.ones((3, height, width))], [[width/2,height/2], [width/2,height/2]]
        out = [get_viewbox(my_path, total_max_diff) for my_path in single_paths]
        viewboxes = [x[0] for x in out]
        centers = [x[1] for x in out]
        if colors is not None:
            rasterized_segments = [raster(disvg(my_path, paths2Drawing=True, colors=[colors[i]], stroke_widths=[stroke_width] * len(my_path), viewbox=viewboxes[i]), out_h = height, out_w = width) for i, my_path in enumerate(single_paths)]
        else:
            rasterized_segments = [raster(disvg(my_path, paths2Drawing=True, stroke_widths=[stroke_width] * len(my_path), viewbox=viewboxes[i]), out_h = height, out_w = width) for i, my_path in enumerate(single_paths)]
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