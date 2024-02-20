import pandas as pd
import os
import glob
from fontTools.ttLib import TTFont
from tqdm import tqdm
from tqdm.auto import tqdm
import argparse
tqdm.pandas()

def svg_path_to_ttf_path(svg_path):
    font_id_parts = svg_path.split("/")[-1].replace(".svg", "").split("_")
    if len(font_id_parts) > 2:
        font_id = "_".join(font_id_parts[1:])
    else:
        font_id = font_id_parts[-1]
    return os.path.join(svg_path.split("/svgs_simplified/")[0], "font_files", font_id+".ttf")

def get_style_of_font(ttf_path):
    font = TTFont(ttf_path)
    # style = font['name'].getName(2, 3, 1).toUnicode()
    style_debug = font["name"].getDebugName(2)
    return style_debug

def simplified_path_to_npy_paths(path:str):
    intermediate_path = path.replace("/svgs_simplified/", "/tokenized/").replace(".svg", ".npy")
    vq_filename = "VQ_" + intermediate_path.split("/")[-1]
    text_filename = "TXT_" + intermediate_path.split("/")[-1]

    vq_path = os.path.join("/".join(intermediate_path.split("/")[:-1]), vq_filename)
    text_path = os.path.join("/".join(intermediate_path.split("/")[:-1]), text_filename)

    return vq_path, text_path

def make_description(row):
    if row['font_style'] is None:
        return f"{'capital ' if row['class'].isupper() else ''}{row['class']}"
    else:
        return f"{'capital ' if row['class'].isupper() else ''}{row['class']} in {row['font_style']} font"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="google_fonts", help="Dataset directory name")
    parser.add_argument("--base_path", default="/scratch2/moritz_data", help="Base path to the dataset")

    args = parser.parse_args()

    DATASET = args.dataset
    BASE_PATH = args.base_path
    OUTPUT_PATH = os.path.join(BASE_PATH, DATASET, "full_split.csv")
    CHECK_FILES_EXISTING = True

    if os.path.exists(OUTPUT_PATH):
        input(f"File {OUTPUT_PATH} already exists. Press enter to overwrite it...")

    # sanity check folder structure
    assert os.path.exists(os.path.join(BASE_PATH, DATASET, "svgs_simplified")), f"Folder {os.path.join(BASE_PATH, DATASET, 'svgs_simplified')} does not exist"
    assert os.path.exists(os.path.join(BASE_PATH, DATASET, "font_files")), f"Folder {os.path.join(BASE_PATH, DATASET, 'font_files')} does not exist"
    assert os.path.exists(os.path.join(BASE_PATH, DATASET, "svgs")), f"Folder {os.path.join(BASE_PATH, DATASET, 'svgs')} does not exist"
    assert os.path.exists(os.path.join(BASE_PATH, DATASET, "tokenized")), f"Folder {os.path.join(BASE_PATH, DATASET, 'tokenized')} does not exist"

    simplified_path = os.path.join(BASE_PATH, DATASET, "svgs_simplified")
    all_svgs = glob.glob(os.path.join(simplified_path, "**/*.svg"), recursive=True)

    input(f"This script will be based on all {len(all_svgs)} simplified SVGs from {simplified_path}. Press enter to continue...")

    df = pd.DataFrame(all_svgs, columns=["simplified_svg_file_path"])
    df["class"] = df["simplified_svg_file_path"].apply(lambda x: x.split("/")[-2])
    df["split"] = df["simplified_svg_file_path"].apply(lambda x: x.split("/")[-3])
    df["font_path"] = df["simplified_svg_file_path"].apply(svg_path_to_ttf_path)
    df["font"] = df["font_path"].apply(lambda x: x.split("/")[-1])
    df["original_svg_file_path"] = df["simplified_svg_file_path"].apply(lambda x: x.replace("/svgs_simplified/", "/svgs/"))

    print("Getting font styles...")
    unique_fonts = df["font"].unique()
    df["font_style"] = ""
    for font in tqdm(unique_fonts):
        style = get_style_of_font(os.path.join(BASE_PATH, DATASET, "font_files", font))
        df.loc[df["font"] == font, "font_style"] = style.lower()

    print("Making descriptions...")
    df["description"] = df.progress_apply(make_description, axis=1)
    
    print("Making token paths...")
    df["vq_token_path"] = df["simplified_svg_file_path"].progress_apply(lambda x: simplified_path_to_npy_paths(x)[0])
    df["text_token_path"] = df["simplified_svg_file_path"].progress_apply(lambda x: simplified_path_to_npy_paths(x)[1])

    if CHECK_FILES_EXISTING:
        all_exist = True
        print("Checking if all files exist...")
        for i, row in df.iterrows():
            if not os.path.exists(row["vq_token_path"]):
                print("[MISSING FILE DETECTED] "+row["vq_token_path"])
                all_exist = False
            if not os.path.exists(row["text_token_path"]):
                print("[MISSING FILE DETECTED] "+row["text_token_path"])
                all_exist = False
            if not os.path.exists(row["simplified_svg_file_path"]):
                print("[MISSING FILE DETECTED] "+row["simplified_svg_file_path"])
                all_exist = False
            if not os.path.exists(row["original_svg_file_path"]):
                print("[MISSING FILE DETECTED] "+row["original_svg_file_path"])
                all_exist = False
            if not os.path.exists(row["font_path"]):
                print("[MISSING FILE DETECTED] "+row["font_path"])
                all_exist = False
        if not all_exist:
            input("Some files are missing. Press enter to continue saving the csv nonetheless...")
        print("Check complete.")

    df.to_csv(OUTPUT_PATH, index=False)
