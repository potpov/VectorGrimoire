import pandas as pd
import os
import glob
from fontTools.ttLib import TTFont
from tqdm import tqdm
from tqdm.auto import tqdm
import argparse
import numpy as np
from concurrent import futures
tqdm.pandas()

def svg_path_to_ttf_path(svg_path):
    font_id_parts = svg_path.split("/")[-1].replace(".svg", "").split("_")
    if len(font_id_parts) > 2:
        font_id = "_".join(font_id_parts[1:])
    else:
        font_id = font_id_parts[-1]
    return os.path.join(svg_path.split("/svgs_simplified/")[0], "font_files", font_id+".ttf")

def get_style_of_font(ttf_path):
    if not os.path.exists(ttf_path):
        return "ERROR: FONT FILE NOT FOUND"
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
    
def make_df(df: pd.DataFrame, out_dir:str, version:int, base_path:str, dataset:str):
    min_index = df.index.start
    max_index = df.index.stop
    save_path = os.path.join(out_dir, f"v{version}_full_split_{min_index}_{max_index}.csv")

    df["class"] = df["simplified_svg_file_path"].apply(lambda x: x.split("/")[-2])
    df["split"] = df["simplified_svg_file_path"].apply(lambda x: x.split("/")[-3])
    df["font_path"] = df["simplified_svg_file_path"].apply(svg_path_to_ttf_path)
    df["font"] = df["font_path"].apply(lambda x: x.split("/")[-1])
    df["original_svg_file_path"] = df["simplified_svg_file_path"].apply(lambda x: x.replace("/svgs_simplified/", "/svgs/"))
    df.to_csv(save_path.replace(".csv", "_temp1.csv"), index=False, escapechar="\\")

    # print("Getting font styles...")
    unique_fonts = df["font"].unique()
    df["font_style"] = ""
    for font in tqdm(unique_fonts):
        style = get_style_of_font(os.path.join(base_path, dataset, "font_files", font))
        if isinstance(style, str):
            df.loc[df["font"] == font, "font_style"] = style.lower()
        else:
            df.loc[df["font"] == font, "font_style"] = "unknown"
    # print("Making descriptions...")
    df["description"] = df.apply(make_description, axis=1)
    
    # print("Making token paths...")
    df["vq_token_path"] = df["simplified_svg_file_path"].apply(lambda x: simplified_path_to_npy_paths(x)[0])
    df["text_token_path"] = df["simplified_svg_file_path"].apply(lambda x: simplified_path_to_npy_paths(x)[1])
    df.to_csv(save_path.replace(".csv", "_temp2.csv"), index=False, escapechar="\\")
    print("Getting token lengths...")
    df["text_token_length"] = df["text_token_path"].progress_apply(lambda x: len(np.load(x)) if os.path.exists(x) else 9999)
    df["vq_token_length"] = df["vq_token_path"].progress_apply(lambda x: len(np.load(x)) if os.path.exists(x) else 9999)
    df.to_csv(save_path, index=False, escapechar="\\")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="google_fonts", help="Dataset directory name")
    parser.add_argument("--base_path", default="/scratch2/moritz_data", help="Base path to the dataset")
    parser.add_argument("--num_workers", default=32, type=int, help="Number of workers to use")
    parser.add_argument("--version", default=1, type=int, help="Version of the dataset")
    parser.add_argument("--type", default="thread", help="Check if all files exist")

    args = parser.parse_args()

    DATASET = args.dataset
    BASE_PATH = args.base_path
    NUM_WORKERS = args.num_workers
    VERSION = args.version
    # CHECK_FILES_EXISTING = False

    # sanity check folder structure
    assert os.path.exists(os.path.join(BASE_PATH, DATASET, "svgs_simplified")), f"Folder {os.path.join(BASE_PATH, DATASET, 'svgs_simplified')} does not exist"
    assert os.path.exists(os.path.join(BASE_PATH, DATASET, "font_files")), f"Folder {os.path.join(BASE_PATH, DATASET, 'font_files')} does not exist"
    assert os.path.exists(os.path.join(BASE_PATH, DATASET, "svgs")), f"Folder {os.path.join(BASE_PATH, DATASET, 'svgs')} does not exist"
    assert os.path.exists(os.path.join(BASE_PATH, DATASET, "tokenized")), f"Folder {os.path.join(BASE_PATH, DATASET, 'tokenized')} does not exist"

    simplified_path = os.path.join(BASE_PATH, DATASET, "svgs_simplified")
    print(f"Searching through {simplified_path} for svgs....")
    all_svgs = glob.glob(os.path.join(simplified_path, "**/*.svg"), recursive=True)

    input(f"This script will be based on all {len(all_svgs)} simplified SVGs from {simplified_path}. Press enter to continue...")

    if args.type == "thread":
        with futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            df = pd.DataFrame(all_svgs, columns=["simplified_svg_file_path"])
            chunk_size = len(df) // NUM_WORKERS
            num_iters = len(df) // chunk_size
            print(f"Making {num_iters} iterations of size {chunk_size} each.")
            input("Press enter to continue...")
            with tqdm(total=num_iters) as pbar:
                    preprocess_requests = [executor.submit(make_df, df.iloc[i*chunk_size:(i+1)*chunk_size].copy(deep=True), os.path.join(BASE_PATH, DATASET), VERSION, BASE_PATH, DATASET)
                                        for i in range(NUM_WORKERS)]

                    for _ in futures.as_completed(preprocess_requests):
                        pbar.update(1)
    elif args.type == "process":
        with futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            df = pd.DataFrame(all_svgs, columns=["simplified_svg_file_path"])
            chunk_size = len(df) // NUM_WORKERS
            num_iters = len(df) // chunk_size
            print(f"Making {num_iters} iterations of size {chunk_size} each.")
            input("Press enter to continue...")
            with tqdm(total=num_iters) as pbar:
                    preprocess_requests = [executor.submit(make_df, df.iloc[i*chunk_size:(i+1)*chunk_size].copy(deep=True), os.path.join(BASE_PATH, DATASET), VERSION, BASE_PATH, DATASET)
                                        for i in range(NUM_WORKERS)]

                    for _ in futures.as_completed(preprocess_requests):
                        pbar.update(1)
    
    # if CHECK_FILES_EXISTING:
    #     all_exist = True
    #     print("Checking if all files exist...")
    #     for i, row in df.iterrows():
    #         if not os.path.exists(row["vq_token_path"]):
    #             print("[MISSING FILE DETECTED] "+row["vq_token_path"])
    #             all_exist = False
    #         if not os.path.exists(row["text_token_path"]):
    #             print("[MISSING FILE DETECTED] "+row["text_token_path"])
    #             all_exist = False
    #         if not os.path.exists(row["simplified_svg_file_path"]):
    #             print("[MISSING FILE DETECTED] "+row["simplified_svg_file_path"])
    #             all_exist = False
    #         if not os.path.exists(row["original_svg_file_path"]):
    #             print("[MISSING FILE DETECTED] "+row["original_svg_file_path"])
    #             all_exist = False
    #         if not os.path.exists(row["font_path"]):
    #             print("[MISSING FILE DETECTED] "+row["font_path"])
    #             all_exist = False
    #     if not all_exist:
    #         df.to_csv(OUTPUT_PATH.replace(".csv", "_temp.csv"), index=False, escapechar="\\")
    #         input("Some files are missing. Press enter to continue saving the csv nonetheless...")
    #     print("Check complete.")

    # df.to_csv(OUTPUT_PATH, index=False, escapechar="\\")
    # os.remove(OUTPUT_PATH.replace(".csv", "_temp.csv"))
