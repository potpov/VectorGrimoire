##
# Used to convert all the generations into PDF for the paper

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import os
import argparse
def explore_folder(folder_path):

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isdir(item_path):
            explore_folder(item_path)

        elif os.path.isfile(item_path) and item.endswith('.svg'):
            try:
                drawing = svg2rlg(item_path)
                renderPDF.drawToFile(drawing, item_path.replace(".svg", ".pdf"))
                os.remove(item_path)
            except:
                print(f"skipping: {item_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get configuration by key.')
    parser.add_argument('--path', type=str, help='Configuration key to use')
    # Parse arguments
    args = parser.parse_args()

    # Retrieve the configuration object using the provided key
    folder_path = args.path

    explore_folder(folder_path)
    print(f"completed")
