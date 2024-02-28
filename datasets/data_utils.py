from tqdm import tqdm
import glob
from svgpathtools import svg2paths, disvg, wsvg
import os

def get_centered_path(path, resolution):
    canvas_height = resolution
    canvas_width = resolution

    curr_x = path.start.real
    curr_y = path.start.imag

    target_x = canvas_width // 2
    target_y = canvas_height // 2

    dx = target_x - curr_x 
    dy = target_y - curr_y

    translation = complex(dx, dy)

    path = path.translated(translation)

    return path

def center_and_save_svg(input_path:str, output_path:str, resolution: int):
    paths, attributes = svg2paths(input_path)
    centered_paths = []
    for path in paths:
        centered_paths.append(get_centered_path(path, resolution))
    wsvg(centered_paths, attributes=attributes, filename=output_path, viewbox=(0,0, resolution, resolution))

def create_centered_dataset(input_path: str, output_path: str, resolution: int, exists_ok = False):
    """
    Args:
        input_path: path to the folder containing the svgs
        output_path: path to the folder where the centered svgs will be saved
        resolution: the resolution of the viewbox for the output svgs
    """
    if os.path.exists(output_path) and not exists_ok:
        print("Output path already exists. Aborting.")
        return
    elif not os.path.exists(output_path):
        os.makedirs(output_path)

    svg_paths = glob.glob(input_path + "/*.svg")
    for i, svg_path in tqdm(enumerate(svg_paths), total=len(svg_paths)):
        output_file = output_path + "/centered_" + str(i) + ".svg"
        center_and_save_svg(svg_path, output_file, resolution)