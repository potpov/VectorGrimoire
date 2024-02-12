from concurrent import futures
import os
from argparse import ArgumentParser
import logging
from tqdm import tqdm
import glob
import pandas as pd
import yaml
from svglib.svg import SVG


def preprocess_svg(char_path):
    svg_files = glob.glob(os.path.join(char_path, "**/*.svg"), recursive=True)
    print(f"Found {len(svg_files)} SVG files.")
    meta_data = {}

    for i, svg_path in enumerate(svg_files):
        if i % (len(svg_files) // 10) == 0:
            print(f"Processed {round((i / len(svg_files)) * 100)}% of {char_path}")
        parts = svg_path.rsplit("/svg/", maxsplit=1)
        new_filename = os.path.join(parts[0], "svg_simplified", parts[1]) if len(parts) > 1 else path
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


def main(config, workers=12):

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

            # df = pd.DataFrame(meta_data.values())
            # if not os.path.exists(os.path.dirname(args.output_meta_file)):
            #     os.makedirs(os.path.dirname(args.output_meta_file))
            # df.to_csv(args.output_meta_file, index=False)

        logging.info(f"SVG Preprocessing complete for dataset: {dataset}.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    with open("font_paths.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    main(config)