from tokenizer import VQTokenizer
from models import VSQ
import numpy as np
import torch
from glob import glob
import os
from tqdm import tqdm
import yaml
import resource

from torch.utils.data import DataLoader
import pandas as pd
import re
from dataset import VSQDataset
import torch.nn as nn
from pathlib import Path

def mycollate(batch):
    imgs, labels, centers, descriptions, filenames = zip(*batch)
    return imgs, labels, centers, descriptions, filenames


def get_existing_Data(tokenizer_dir):
    existing = []
    for _split in ["train", "test"]:
        for folder in os.listdir(os.path.join(tokenizer_dir, _split)):
            existing += [f for f in os.listdir(os.path.join(tokenizer_dir, _split, folder)) if "TXT" not in f]  # just VQ
    existing = [e.replace("VQ_", "").replace("npy", "svg") for e in existing]
    return existing


class SkipDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, existing):
        self.original_dataset = original_dataset
        full_split = self.original_dataset.split
        before_len = len(full_split)
        print(f"original split len: {before_len}")
        basenames = full_split['file_path'].apply(os.path.basename)
        self.original_dataset.split = full_split[~basenames.isin(existing)]  # overriding
        print(f"we removed {before_len - len(self.original_dataset.split)} elements which were already preprocessed")

    def __getitem__(self, index):
        return self.original_dataset[index]

    def __len__(self):
        return len(self.original_dataset)


def main():

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    # MODEL_WEIGHTS_PATH = "/scratch/mcipriano/results/svg/all_full_single_code/checkpoints/last-v1.ckpt"
    # MODEL_WEIGHTS_PATH = "/scratch/mcipriano/results/svg/figr8/checkpoints/epoch=0-step=10500.ckpt"
    MODEL_WEIGHTS_PATH = "/scratch/mcipriano/cache/svg/moritz_geometric.ckpt"
    CONFIG_PATH = "/home/mcipriano/projects/SVG/Moritz/configs/SVG_VQVAE_Stage1.yaml"
    BATCH_SIZE = 16

    with open("font_paths.yaml", "r") as stream:
        font_config = yaml.safe_load(stream)

    for dataset, params in font_config["fonts"].items():
        print(f"Processing {dataset}")

        with open(CONFIG_PATH, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        config['data_params']["max_shapes_per_svg"] = 512  # more than context length of 1024 (at least one patch and one pos token per svg shape) we'll never do I think

        Path(os.path.join(params["tokenized"])).mkdir(parents=True, exist_ok=True)
        # Path(os.path.join(params["tokenized"], "train")).mkdir(parents=True, exist_ok=True)
        # Path(os.path.join(params["tokenized"], "test")).mkdir(parents=True, exist_ok=True)
        # if len(os.listdir(os.path.join(params["tokenized"], "train"))) > 0:
        #     print("Output directory is not empty, found: ", os.listdir(params["tokenized"]))
        #     input("Press Enter to continue...")

        #################
        ###  MODEL
        print("Loading model..")
        device = torch.device("cuda")
        model = VSQ(**config['model_params'])
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)["state_dict"]
        try:
            model.load_state_dict(state_dict)
        except:
            model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})

        model = model.eval()
        model = model.to(device)
        tokenizer = VQTokenizer(model, config["data_params"]["width"], 1, "bert-base-uncased")

        print("Loading dataset..")
        # existing_data = get_existing_Data(params["tokenized"])
        existing_data = []

        config['data_params']["top_level_dir"] = params["svg_simp"]
        print("loading from: ", config['data_params']["top_level_dir"])
        ds_train = SkipDataset(
            VSQDataset(train=True, **config['data_params'], return_filename=True),
            existing_data
        )
        ds_test = SkipDataset(
            VSQDataset(train=False, **config['data_params'], return_filename=True),
            existing_data
        )

        dl_train = DataLoader(
            ds_train,
            batch_size=BATCH_SIZE,
            num_workers=8,
            shuffle=False,
            pin_memory=False,
            collate_fn=mycollate,
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=BATCH_SIZE,
            num_workers=8,
            shuffle=False,
            pin_memory=False,
            collate_fn=mycollate,
        )
        print("Number of Tokens: ", tokenizer.num_tokens)

        print("Processing training set..")

        save_csv = {"index_in_numpy_array": [], "class": [], "split": []}
        numpy_arrays = []
        numpy_counter = 0
        for split_name, split in {"train": dl_train, "test": dl_test}.items():
            for i, batch in tqdm(enumerate(split), total=len(split), desc=f"processing {split_name}"):
                imgs, labels, centers, descriptions, filenames = batch
                for img, label, center, description, filename in zip(imgs, labels, centers, descriptions, filenames):
                    start_token, text_tokens, vq_tokens, end_token = tokenizer.tokenize(img.to(device), center, text=description, return_np_uint16=True)
                    class_id, name = filename.split("/")[-2:]
                    save_csv["index_in_numpy_array"].append(numpy_counter)
                    save_csv["class"].append(class_id)
                    save_csv["split"].append(split_name)
                    numpy_arrays.append(vq_tokens)
                    numpy_counter += 1

        np.save(os.path.join(params["tokenized"], "tokenized.npy"), np.concatenate(numpy_arrays))
        df = pd.DataFrame(save_csv)
        df.to_csv(os.path.join(params["tokenized"], "split.csv"), index=False)

if __name__ == '__main__':
    # print(get_latest_data_checkpoint("/scratch2/moritz_data/glyphazzn"))
    main()