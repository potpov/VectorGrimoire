import numpy as np
import torch
from glob import glob
import os
from tqdm import tqdm
import yaml
import shutil
import resource
from pathlib import Path
from torch.utils.data import DataLoader
import pandas as pd
import re
import torch.nn as nn

from dataset import VSQDataset
from tokenizer import VQTokenizer
from models import VSQ
from utils import map_wand_config


def mycollate(batch):
    imgs, labels, centers, descriptions, filenames = zip(*batch)
    return imgs, labels, centers, descriptions, filenames


def get_existing_Data(tokenizer_dir):
    existing = []
    for _split in ["train", "test"]:
        for folder in os.listdir(os.path.join(tokenizer_dir, _split)):
            existing += [f for f in os.listdir(os.path.join(tokenizer_dir, _split, folder)) if
                         "TXT" not in f]  # just VQ
    existing = [e.replace("VQ_", "").replace("npy", "svg") for e in existing]
    return existing


class SkipDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, existing):
        self.original_dataset = original_dataset
        full_split = self.original_dataset.df
        before_len = len(full_split)
        print(f"original split len: {before_len}")
        print(full_split.columns)
        basenames = full_split['simplified_svg_file_path'].apply(os.path.basename)
        self.original_dataset.df = full_split[~basenames.isin(existing)]  # overriding
        print(f"we removed {before_len - len(self.original_dataset.df)} elements which were already preprocessed")

    def __getitem__(self, index):
        return self.original_dataset[index]

    def __len__(self):
        return len(self.original_dataset)


def main():
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    #### CONFIGURATION FOR ICONS
    # MODEL_WEIGHTS_PATH = "/raid/marco.cipriano/results/svg/Grimoire/VSQ/figr8/VSQ_nseg=4_ncode=2_lseg=5.0_alpha=0.1None/checkpoints/last.ckpt"
    # CONFIG_PATH = "/scratch2/moritz_logs/thesis/VSQ_figr8/VSQ_nseg=4_ncode=2_lseg=5.0_alpha=0.1None/config.yaml"
    # OUT_BASE_DIR = "/scratch2/moritz_data/figr8/tokenized_256grid"

    #### CONFIGURATION FOR FONTS
    MODEL_WEIGHTS_PATH = "/raid/marco.cipriano/results/svg/Grimoire/VSQ/VSQ_FONTS/checkpoints/last.ckpt"
    CONFIG_PATH = "/raid/marco.cipriano/results/svg/Grimoire/VSQ/VSQ_FONTS/config.yaml"
    OUT_BASE_DIR = "/raid/marco.cipriano/data/SVG/Grimoire/fonts/tokenized/"

    BATCH_SIZE = 8
    POSITION_GRID_SIZE = 256

    with open(CONFIG_PATH, 'r') as file:
        try:
            config = yaml.safe_load(file)
            config = map_wand_config(config)
        except yaml.YAMLError as exc:
            print(exc)

    out_dir = os.path.join(OUT_BASE_DIR, config["logging_params"]["name"])
    if os.path.exists(os.path.join(out_dir, "tokenized.npy")):
        input(f"{out_dir} already carries a tokenized numpy file, want to overwrite? Press enter...")
        shutil.copy2(os.path.join(out_dir, "tokenized.npy"), os.path.join(out_dir, "backup_tokenized.npy"))

    config['data_params'][
        "max_shapes_per_svg"] = 512  # more than context length of 1024 (at least one patch and one pos token per svg shape) we'll never do I think
    individual_max_length = config["data_params"]["individual_max_length"]
    stroke_width = (float(individual_max_length) + 2) * 0.4 / 5.0
    config["data_params"]["stroke_width"] = stroke_width

    os.makedirs(out_dir, exist_ok=True)

    #################
    ###  MODEL
    print("Loading model..")
    device = torch.device("cuda")
    model = VSQ(patch_size=config["data_params"]["width"], **config['model_params'])
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except:
        model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})

    model = model.eval()
    model = model.to(device)
    tokenizer = VQTokenizer(model, POSITION_GRID_SIZE, config["model_params"]["num_codes_per_shape"],
                            "google/bert_uncased_L-12_H-512_A-8", device=device)

    print("Loading dataset..")
    # existing_data = get_existing_Data(params["tokenized"])
    existing_data = []

    ds_train = SkipDataset(
        VSQDataset(train=True, **config['data_params'], return_filename=True),
        existing_data
    )
    ds_val = SkipDataset(
        VSQDataset(train=False, **config['data_params'], return_filename=True),
        existing_data
    )
    ds_test = SkipDataset(
        VSQDataset(train=None, **config['data_params'], return_filename=True),
        existing_data
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        num_workers=16,
        shuffle=False,
        pin_memory=False,
        collate_fn=mycollate,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        num_workers=16,
        shuffle=False,
        pin_memory=False,
        collate_fn=mycollate,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=BATCH_SIZE,
        num_workers=16,
        shuffle=False,
        pin_memory=False,
        collate_fn=mycollate,
    )
    print("Number of Tokens: ", tokenizer.num_tokens)

    print("Processing training set..")

    save_csv = {"index_in_numpy_array": [],
                "class": [],
                "split": [],
                "filename": [],
                "iconshop_description": [],
                "id": [],
                "vq_token_length": [],
                "text_token_length": [],
                "description": []}
    numpy_arrays = []
    numpy_counter = 0
    for split_name, split in {"train": dl_train, "test": dl_test, "val": dl_val}.items():
        reference_df = split.dataset.original_dataset.df.copy(deep=True)
        reference_df.index = reference_df["id"]
        for i, batch in tqdm(enumerate(split), total=len(split), desc=f"processing {split_name}"):
            imgs, labels, centers, descriptions, filenames = batch
            for img, label, center, description, filename in zip(imgs, labels, centers, descriptions, filenames):
                # TODO maybe this is wrong here
                center = center * tokenizer.full_image_res
                start_token, text_tokens, vq_tokens, end_token = tokenizer.tokenize(img.to(device), center,
                                                                                    text=description,
                                                                                    return_np_uint16=tokenizer.num_tokens < 2 ** 16)
                if isinstance(text_tokens, torch.Tensor):
                    text_tokens = text_tokens.cpu().numpy()
                if isinstance(vq_tokens, torch.Tensor):
                    vq_tokens = vq_tokens.cpu().numpy()
                class_id, name = filename.split("/")[-2:]
                curr_id = name.replace(".svg", "")
                save_csv["filename"].append(filename)
                save_csv["index_in_numpy_array"].append(numpy_counter)
                save_csv["class"].append(class_id)
                save_csv["iconshop_description"].append(description)
                save_csv["split"].append(split_name)
                save_csv["id"].append(curr_id)
                save_csv["vq_token_length"].append(len(vq_tokens))
                numpy_arrays.append(vq_tokens)
                numpy_counter += 1
                save_csv["text_token_length"].append(len(text_tokens))
                # if i==0:
                #     print(vq_tokens)
                try:
                    keyword_description = reference_df.loc[curr_id].label.replace("/", " ")
                except:
                    keyword_description = class_id
                    print("encountered error in getting description")
                save_csv["description"].append(keyword_description)
                assert len(
                    numpy_arrays) == numpy_counter, f"numpy_counter: {numpy_counter}, len(numpy_arrays): {len(numpy_arrays)}"
            if i % 100 == 0:
                np.save(os.path.join(out_dir, "tokenized.npy"), np.concatenate(numpy_arrays))
                df = pd.DataFrame(save_csv)
                df.to_csv(os.path.join(out_dir, "stage2_split.csv"), index=False)
        np.save(os.path.join(out_dir, "tokenized.npy"), np.concatenate(numpy_arrays))
        df = pd.DataFrame(save_csv)
        df.to_csv(os.path.join(out_dir, "stage2_split.csv"), index=False)
    np.save(os.path.join(out_dir, "tokenized.npy"), np.concatenate(numpy_arrays))
    df = pd.DataFrame(save_csv)
    df.to_csv(os.path.join(out_dir, "stage2_split.csv"), index=False)


if __name__ == '__main__':
    # print(get_latest_data_checkpoint("/scratch2/moritz_data/glyphazzn"))
    main()