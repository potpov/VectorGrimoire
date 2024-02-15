import random
from concurrent import futures
import os
from argparse import ArgumentParser
import logging
from tqdm import tqdm
import glob
import pandas as pd
import yaml
from svglib.svg import SVG
from fontTools.ttLib import TTFont
import ast


def preprocess_svg(char_path):
    svg_files = glob.glob(os.path.join(char_path, "**/*.svg"), recursive=True)
    print(f"Found {len(svg_files)} SVG files.")
    meta_data = {}

    for i, svg_path in enumerate(svg_files):
        if i % (len(svg_files) // 10) == 0:
            print(f"Processed {round((i / len(svg_files)) * 100)}% of {char_path}")
        parts = svg_path.rsplit("/svg/", maxsplit=1)
        new_filename = os.path.join(parts[0], "svg_simplified", parts[1]) if len(parts) > 1 else svg_path
        if not os.path.exists(os.path.dirname(new_filename)):
            os.makedirs(os.path.dirname(new_filename))
        try:
            if os.path.exists(new_filename):
                continue
            svg = SVG.load_svg(svg_path)
            svg.fill_(False)
            svg.normalize()
            svg.zoom(0.9)
            svg.canonicalize()
            svg = svg.simplify_heuristic(epsilon=0.001)

            len_groups = [path_group.total_len() for path_group in svg.svg_path_groups]

            meta_data[new_filename] = {
                "id": new_filename.split("/")[-1].split(".")[0],
                "path": new_filename,
                "total_len": sum(len_groups),
                "nb_groups": len(len_groups),
                "len_groups": len_groups,
                "max_len_group": max(len_groups)
            }
            svg.save_svg(new_filename)

        except Exception as e:
            print(f"Error processing {svg_path}: {e}")
            if isinstance(e, KeyboardInterrupt):
                raise e

    try:
        df = pd.DataFrame(meta_data.values())
        save_path = os.path.join(char_path.replace("/svg/", "/svg_simplified/"), "meta_data.csv")
        df.to_csv(save_path, index=False)
    except:
        print(f"Error saving meta_data.csv for {char_path}")


def multi_thread_svg_preprocess(config, workers=12):
    for dataset, params in config["fonts"].items():
        # each split is handled separately
        # we divide the work into processes, one per character
        for split in ["train", "test"]:
            print(f"Processing {split} split for {dataset}.")
            curr_data_folder = os.path.join(params["svg_dir"], split)
            with futures.ProcessPoolExecutor(max_workers=workers) as executor:
                all_dirs = glob.glob(os.path.join(curr_data_folder, "*"))
                all_dirs = [d for d in all_dirs if os.path.isdir(d)]

                with tqdm(total=len(all_dirs)) as pbar:
                    preprocess_requests = [executor.submit(preprocess_svg, char_dir)
                                           for char_dir in all_dirs]

                    for _ in futures.as_completed(preprocess_requests):
                        pbar.update(1)

        logging.info(f"SVG Preprocessing complete for dataset: {dataset}.")


def process_csv_file(csv_filename, metadata, params):
    df = pd.read_csv(os.path.join(params["svg_dir"], csv_filename))
    print(f"Processing {os.path.join(params['svg_dir'], csv_filename)}, with {len(df)} rows")

    new_csv = pd.DataFrame()
    for index, row in df.iterrows():
        file_path = row['output_path']
        assert params["svg_dir"] in file_path, f"This is a very weird csv path: {file_path}"
        file_path = file_path.replace(params["svg_dir"], params["svg_simp"])

        if os.path.exists(file_path):  # check if simplified svg exist
            font_path = row['font_path']
            character = row['char']

            font = TTFont(font_path)
            style = font["name"].getDebugName(2) if font["name"].getDebugName(2) is not None else "undefined"
            descrition = f"{character} in a {style} font"

            font_name = row['font_name']
            if metadata is not None:
                font_class = metadata[metadata['filename'].str.contains(font_name.lower())]
                if len(font_class) > 0:
                    classes = ast.literal_eval(metadata[metadata['filename'].str.contains(font_name.lower())].tags.iloc[0])
                    style = random.choice(classes)
                    descrition = f"{character} in a {', '.join(classes)} font"

            new_row = pd.DataFrame({
                "file_path": [file_path],
                "class": [character],
                "split": [row['split']],
                "font_path": [font_path],
                "font": [font_name],
                "font_style": [style],
                "description": [descrition],
            })

            new_csv = pd.concat([new_csv, new_row], ignore_index=True)
    new_csv.to_csv(os.path.join(params["svg_simp"], csv_filename), index=False)


def multi_thread_csv_preprocess(config):
    for dataset, params in config["fonts"].items():
        print(f"processing {dataset}.")
        csv_shards = [f for f in os.listdir(params["svg_dir"]) if f.endswith(".csv")]

        metadata = None
        if os.path.exists(os.path.join(params["fonts_dir"], "metadata.csv")):
            metadata = pd.read_csv(os.path.join(params["fonts_dir"], "metadata.csv"))

        with futures.ProcessPoolExecutor(max_workers=len(csv_shards)) as executor:
            with tqdm(total=len(csv_shards)) as pbar:
                preprocess_requests = [executor.submit(process_csv_file, csv_name, metadata, params)
                                       for csv_name in csv_shards]
                for _ in futures.as_completed(preprocess_requests):
                    pbar.update(1)

        print("all shards processed. Creating one giant split file by merging them")
        # merging up results
        csv_files = [f for f in os.listdir(params["svg_simp"]) if f.endswith(".csv")]
        merged_df = pd.concat([pd.read_csv(os.path.join(params["svg_simp"], csv_file)) for csv_file in csv_files], ignore_index=True)
        merged_df.to_csv(os.path.join(params["svg_simp"], "split.csv"), index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    with open("font_paths.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    multi_thread_csv_preprocess(config)
    # multi_thread_svg_preprocess(config)