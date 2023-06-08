import os
import glob
import cairosvg
from tqdm import tqdm

TARGET_RESOLUTION = 512
OUTPUT_DIR = "/scratch1/emoji_png/full"
PATH_TO_TWEMOJI = None # e.g. /scratch1/twemoji

if(__name__=="__main__"):
    if(PATH_TO_TWEMOJI is None):
        raise FileNotFoundError("You need to clone the twemoji repository first and specify the path in the script")

    svg_path = os.path.join(PATH_TO_TWEMOJI, "assets", "svg")
    all_files = glob.glob(svg_path+"/*.svg")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in tqdm(all_files):
        file_name = path.split("/")[-1]
        new_file_name = file_name.replace(".svg", ".png")
        cairosvg.svg2png(url=path, write_to=os.path.join(OUTPUT_DIR, new_file_name), output_width=TARGET_RESOLUTION, output_height=TARGET_RESOLUTION, background_color="white")
