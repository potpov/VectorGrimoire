from fontTools.ttLib import TTFont
import os
from glob import glob

SCRAPE_DIR = "/scratch/datasets/svg/scraped_fonts"


def convert_to_ttf(input_file, output_file):
    try:
        font = TTFont(input_file)
        font.save(output_file)
        print(f"Converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error converting {input_file}: {e}")


def convert_folder_to_ttf():
    extensions = ['otf', 'eot', 'woff', 'woff2']
    for extension in extensions:
        font_files = glob(os.path.join(SCRAPE_DIR, f'**/*.{extension}'), recursive=True)
        for font_file in font_files:
                    super_dir = os.path.dirname(font_file)
                    output_path = os.path.join(super_dir, f"{os.path.splitext(os.path.basename(font_file))[0]}.ttf")
                    # Check if the TTF file already exists, and skip if it does
                    if not os.path.exists(output_path):
                        convert_to_ttf(font_file, output_path)
                    else:
                        print(f"TTF file {output_path} already exists. Skipping.")


if __name__ == '__main__':
    convert_folder_to_ttf()
