from models import Vector_VQVAE
from tokenizer import RasterVQTokenizer
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
from dataset import TiledMNIST, MNISTDataset
import torch.nn as nn
from pathlib import Path


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
    # MODEL_WEIGHTS_PATH = "/scratch/mcipriano/cache/svg/moritz_geometric.ckpt"
    MODEL_WEIGHTS_PATH = None
    CONFIG_PATH = "/home/moe/GitHub/thesis/configs/VSQ_mnist_local.yaml"
    OUT_PATH = "/home/moe/GitHub/thesis/.data/tokenized_mnist"
    BATCH_SIZE = 16

    with open(CONFIG_PATH, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
    # Path(os.path.join(params["tokenized"], "train")).mkdir(parents=True, exist_ok=True)
    # Path(os.path.join(params["tokenized"], "test")).mkdir(parents=True, exist_ok=True)
    # if len(os.listdir(os.path.join(params["tokenized"], "train"))) > 0:
    #     print("Output directory is not empty, found: ", os.listdir(params["tokenized"]))
    #     input("Press Enter to continue...")

    #################
    ###  MODEL
    print("Loading model..")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = Vector_VQVAE(patch_size = config['data_params']["patch_size"], **config['model_params'])
    if MODEL_WEIGHTS_PATH is not None:
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)["state_dict"]
        try:
            model.load_state_dict(state_dict)
        except:
            model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})

    model = model.eval()
    model = model.to(device)
    tokenizer = RasterVQTokenizer(model, 
                                  tokens_per_patch=1, 
                                  do_tokenize_positions=False,
                                  patch_size=config['data_params']["patch_size"],
                                  num_tiles_per_row=config['data_params']["num_tiles_per_row"],)

    print("Loading dataset..")
    config['data_params']["train_batch_size"] = 1
    config['data_params']["test_batch_size"] = 1
    config['data_params']["val_batch_size"] = 1
    datamodule = MNISTDataset(**config['data_params'], return_filename=True)
    datamodule.setup()

    dl_train = datamodule.train_dataloader()
    dl_test = datamodule.test_dataloader()

    print("Number of Tokens: ", tokenizer.num_tokens)

    print("Processing training set..")

    save_csv = {"index_in_numpy_array": [], "filename": [], "split": [], "label": []}
    numpy_arrays = []
    numpy_counter = 0
    for split_name, split in {"train": dl_train, "test": dl_test}.items():
        for i, batch in tqdm(enumerate(split), total=len(split), desc=f"processing {split_name}"):
            imgs, labels, _, descriptions, filenames = batch
            # print(imgs, labels, centers, descriptions, filenames)
            
            print(imgs.shape, descriptions, filenames)
            start_token, text_tokens, vq_tokens, end_token = tokenizer.tokenize(imgs.to(device), text=descriptions[0], return_np_uint16=True)
            save_csv["index_in_numpy_array"].append(numpy_counter)
            save_csv["filename"].append(filenames[0])
            save_csv["split"].append(split_name)
            save_csv["label"].append(labels[0])
            # print("vq_tokens: ",vq_tokens)
            numpy_arrays.append(vq_tokens)
            numpy_counter += 1
            if numpy_counter % 200 == 0:
                print(numpy_counter)

    np.save(os.path.join(OUT_PATH, "tokenized.npy"), np.concatenate(numpy_arrays))
    df = pd.DataFrame(save_csv)
    df.to_csv(os.path.join(OUT_PATH, "split.csv"), index=False)

if __name__ == '__main__':
    # print(get_latest_data_checkpoint("/scratch2/moritz_data/glyphazzn"))
    main()