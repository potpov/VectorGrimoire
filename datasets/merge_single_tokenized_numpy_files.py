import pandas as pd
import os
import glob
from fontTools.ttLib import TTFont
from tqdm import tqdm
from tqdm.auto import tqdm
import argparse
import numpy as np
from concurrent import futures
import time
import shutil


tqdm.pandas()

def merge_csv_files(save_dir, output_csv_name):
    # Updated the glob pattern to match the new file naming scheme
    csv_files = sorted(
        glob.glob(os.path.join(save_dir, "glyphazzn_allfreefonts_dafont_merged_filtered_v1_fixed_start_idx_*.csv")),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    all_dfs = [pd.read_csv(f) for f in csv_files]
    full_df = pd.concat(all_dfs).sort_values(by="original_idx").reset_index(drop=True)
    full_df.to_csv(os.path.join(save_dir, output_csv_name), index=False)
    return csv_files  # Return the list of files for moving them later

def merge_npy_files(save_dir, output_npy_name):
    # Updated the glob pattern to match the new file naming scheme and included save_dir in path
    npy_files = sorted(
        glob.glob(os.path.join(save_dir, "vq_tokens_*.npy")),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    all_arrays = [np.load(f) for f in npy_files]
    full_array = np.concatenate(all_arrays)
    np.save(os.path.join(save_dir, output_npy_name), full_array)
    return npy_files  # Return the list of files for moving them later

def move_processed_files(files, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for f in files:
        shutil.move(f, target_dir)

def make_df(df: pd.DataFrame, save_dir: str):
    all_vq_tokens = []
    start_idx = df.index[0]

    for i in tqdm(range(start_idx, start_idx + len(df)), total=len(df)):
        try:
            all_vq_tokens.append(np.load(df.loc[i]["vq_token_path"]))
            df.loc[i, "index_in_numpy_array"] = i
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            print(e)
            pass
    df.to_csv(os.path.join(save_dir, f"glyphazzn_allfreefonts_dafont_merged_filtered_v1_fixed_start_idx_{start_idx}.csv"), index=True, index_label="original_idx")
    np.save(os.path.join(save_dir, f"vq_tokens_{start_idx}.npy"), np.concatenate(all_vq_tokens))

if __name__ == "__main__":
    SAVE_DIR = "/scratch2/moritz_data/full_fonts/merged_csv"
    NUM_WORKERS = 32

    args = argparse.ArgumentParser()
    args.add_argument("--type", type=str, default="thread")
    args = args.parse_args()
    start_time = time.time()

    if args.type == "thread":
        with futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            full_df = pd.read_csv("/scratch2/moritz_data/full_fonts/glyphazzn_allfreefonts_dafont_merged_filtered_v1.csv")
            full_df = full_df[full_df["text_token_length"] < 32]
            full_df = full_df[full_df["vq_token_length"] < 9999]
            full_df = full_df[full_df["vq_token_path"].apply(lambda x: os.path.exists(x))]
            full_df = full_df.reset_index(drop=True)
            full_df.to_csv(os.path.join(SAVE_DIR, "glyphazzn_allfreefonts_dafont_merged_filtered_v1_fixed.csv"), index=False)


            chunk_size = len(full_df) // NUM_WORKERS
            num_iters = len(full_df) // chunk_size
            print(f"Making {num_iters} iterations of size {chunk_size} each.")
            input("Press enter to continue...")
            with tqdm(total=num_iters) as pbar:
                    preprocess_requests = [executor.submit(make_df, full_df.iloc[i*chunk_size:(i+1)*chunk_size].copy(deep=True), SAVE_DIR)
                                        for i in range(NUM_WORKERS)]

                    for _ in futures.as_completed(preprocess_requests):
                        pbar.update(1)
    elif args.type == "process":
        with futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            full_df = pd.read_csv("/scratch2/moritz_data/full_fonts/glyphazzn_allfreefonts_dafont_merged_filtered_v1.csv")
            full_df = full_df[full_df["text_token_length"] < 32]
            full_df = full_df[full_df["vq_token_length"] < 9999]
            full_df = full_df[full_df["vq_token_path"].apply(lambda x: os.path.exists(x))]
            full_df = full_df.reset_index(drop=True)
            full_df.to_csv(os.path.join(SAVE_DIR, "glyphazzn_allfreefonts_dafont_merged_filtered_v1_fixed.csv"), index=False)


            chunk_size = len(full_df) // NUM_WORKERS
            num_iters = len(full_df) // chunk_size
            print(f"Making {num_iters} iterations of size {chunk_size} each.")
            input("Press enter to continue...")
            with tqdm(total=num_iters) as pbar:
                    preprocess_requests = [executor.submit(make_df, full_df.iloc[i*chunk_size:(i+1)*chunk_size].copy(deep=True), SAVE_DIR)
                                        for i in range(NUM_WORKERS)]

                    for _ in futures.as_completed(preprocess_requests):
                        pbar.update(1)
    
    print(f"Finished in {round((time.time() - start_time) / 60)} minutes.")
    USED_FILES_DIR = os.path.join(SAVE_DIR, "used_files")  # Define the directory for used files

    if os.path.exists(os.path.join(SAVE_DIR, "combined_full_fonts_fixed.csv")):
        input("The combined CSV file already exists. Press enter to override...")

    # Merge and save CSV files
    csv_files = merge_csv_files(SAVE_DIR, "combined_full_fonts_fixed.csv")
    # Merge and save NPY files
    npy_files = merge_npy_files(SAVE_DIR, "combined_vq_tokens_fixed.npy")
    
    # Move processed files to "used_files" directory
    move_processed_files(csv_files + npy_files, USED_FILES_DIR)

    print("Merging and cleanup completed successfully.")

