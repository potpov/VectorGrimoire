from tokenizer import VQTokenizer
from dataset import CenterShapeLayersFromSVGDataset
from models import Vector_VQVAE
import numpy as np
import torch
from glob import glob
import os
from tqdm import tqdm
import yaml
import pandas as pd
import time
from torch.utils.data import DataLoader
import re

def mycollate(batch):
    imgs, labels, centers, descriptions = zip(*batch)
    return imgs, labels, centers, descriptions

def get_latest_data_checkpoint(dir):
    existing_files = os.listdir(dir)
    file_string = " ".join(existing_files)

    try:
        train_vq_checkpoint = sorted([int(x) for x in re.findall(r"train_vq_tokens_(\d+)\.npy", file_string)])[-1]
    except IndexError:
        train_vq_checkpoint = 0
    try:
        train_text_checkpoint = sorted([int(x) for x in re.findall(r"train_text_tokens_(\d+)\.npy", file_string)])[-1]
    except IndexError:
        train_text_checkpoint = 0
    try:
        test_vq_checkpoint = sorted([int(x) for x in re.findall(r"test_vq_tokens_(\d+)\.npy", file_string)])[-1]
    except IndexError:
        test_vq_checkpoint = 0
    try:
        test_text_checkpoint = sorted([int(x) for x in re.findall(r"test_text_tokens_(\d+)\.npy", file_string)])[-1]
    except IndexError:
        test_text_checkpoint = 0

    assert test_text_checkpoint == test_vq_checkpoint, f"Test vq ({test_vq_checkpoint}) and text tokens ({test_text_checkpoint}) are not in sync"
    assert train_text_checkpoint == train_vq_checkpoint, f"Train vq ({train_vq_checkpoint}) and text tokens ({train_text_checkpoint}) are not in sync"

    return train_vq_checkpoint, test_vq_checkpoint

class SkipDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, start_index):
        self.original_dataset = original_dataset
        self.start_index = start_index

    def __getitem__(self, index):
        return self.original_dataset[index + self.start_index]

    def __len__(self):
        return len(self.original_dataset) - self.start_index

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    CONTEXT_LENGTH = 256
    MODEL_WEIGHTS_PATH = "/scratch2/moritz_logs/SVG_VQVAE/Stage1/glyphazzn_full_single_code/checkpoints/last.ckpt"
    OUT_DIR = "/scratch2/moritz_data/glyphazzn/tokenized"
    TOP_LEVEL_DIR = "/scratch2/moritz_data/glyphazzn/svgs_simplified"
    CONFIG_PATH = "/home/mfeuerpfeil/master/thesis/configs/SVG_VQVAE_Stage1.yaml"
    BATCH_SIZE = 16

    with open(CONFIG_PATH, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config['data_params']["max_shapes_per_svg"] = 512  # more than context length of 1024 (at least one patch and one pos token per svg shape) we'll never do I think
        
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    if len(os.listdir(OUT_DIR)) > 0:
        print("Output directory is not empty, found: ", os.listdir(OUT_DIR))
        input("Press Enter to continue...")
    
    
    print("Loading model..")
    start_time = time.time()
    model = Vector_VQVAE(**config['model_params']).to(device)
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except:
        model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
    model = model.eval()
    elapsed_time = time.time() - start_time
    print(f"Model loaded in {elapsed_time} seconds")

    start_time = time.time()
    tokenizer = VQTokenizer(model, config["data_params"]["width"], 1, "bert-base-uncased")
    elapsed_time = time.time() - start_time
    print(f"Tokenizer loaded in {elapsed_time} seconds")

    start_time = time.time()
    print("Loading dataset..")

    train_checkpoint, test_checkpoint = get_latest_data_checkpoint(OUT_DIR)

    ds_train = SkipDataset(GlyphazznStage1Dataset(train=True, **config['data_params']), start_index = train_checkpoint)
    ds_test = SkipDataset(GlyphazznStage1Dataset(train=False, **config['data_params']), start_index = test_checkpoint)
    print(f"Starting from datapoint {train_checkpoint} for train and {test_checkpoint} for test.")
    print(f"That is {round(train_checkpoint / (len(ds_train) + train_checkpoint), 2)} of the train dataset and {round(test_checkpoint / (len(ds_test) + test_checkpoint), 2)} of the test dataset.")
    dl_train = DataLoader(ds_train,
        batch_size=BATCH_SIZE,
        num_workers=16,
        shuffle=False,
        pin_memory=False,
        collate_fn=mycollate,
    )
    dl_test = DataLoader(ds_test,
        batch_size=BATCH_SIZE,
        num_workers=16,
        shuffle=False,
        pin_memory=False,
        collate_fn=mycollate,
    )

    if "train_vq_tokens_last.npy" in os.listdir(OUT_DIR) and "train_text_tokens_last.npy" in os.listdir(OUT_DIR):
        print(f"NVM, Found train_vq_tokens_last.npy in {OUT_DIR}, skipping generating training data..")
        dl_train = []
    if "test_vq_tokens_last.npy" in os.listdir(OUT_DIR) and "test_text_tokens_last" in os.listdir(OUT_DIR):
        print(f"NVM, Found test_vq_tokens_last.npy in {OUT_DIR}, skipping generating test data..")
        dl_test = []

    elapsed_time = time.time() - start_time
    print(f"Datasets loaded in {elapsed_time} seconds")
    print("Number of Tokens: ",tokenizer.num_tokens)

    train_vq_tokens = []
    train_text_tokens = []
    get_item_times = []
    tokenization_times = []
    input("Press Enter to continue...")

    print("Processing training set..")
    for i, batch in tqdm(enumerate(dl_train), total=len(dl_train)):
        start_time = time.time()
        imgs, labels, centers, descriptions = batch
        elapsed_time = time.time() - start_time
        get_item_times.append(elapsed_time)
        start_time = time.time()
        for img, label, center, description in zip(imgs, labels, centers, descriptions):
            start_token, text_tokens, vq_tokens, end_token = tokenizer.tokenize(img.to(device), center, text=description, return_np_uint16=True)
            train_vq_tokens.append(vq_tokens)
            train_text_tokens.append(text_tokens)
        elapsed_time = time.time() - start_time
        tokenization_times.append(elapsed_time)
        if i % 100 == 0 and i < 1000:
            print(f"Average dataloader time: {round(sum(get_item_times) / len(get_item_times), 2)} seconds")
            print(f"Average tokenization time: {round(sum(tokenization_times) / len(tokenization_times), 2)} seconds")
        if i % 2000 == 0 and i > 0:
            np.save(os.path.join(OUT_DIR, f"train_vq_tokens_{i*BATCH_SIZE + train_checkpoint}.npy"), np.concatenate(train_vq_tokens))
            np.save(os.path.join(OUT_DIR, f"train_text_tokens_{i*BATCH_SIZE + train_checkpoint}.npy"), np.concatenate(train_text_tokens))
            train_vq_tokens=[]
            train_text_tokens=[]
    if len(train_vq_tokens) > 0 and len(train_text_tokens) > 0:
        train_vq_tokens = np.concatenate(train_vq_tokens)
        train_text_tokens = np.concatenate(train_text_tokens)
        np.save(os.path.join(OUT_DIR, "train_vq_tokens_last.npy"), train_vq_tokens)
        np.save(os.path.join(OUT_DIR, "train_text_tokens_last.npy"), train_text_tokens)

    test_vq_tokens = []
    test_text_tokens = []
    print("Processing test set..")
    for i, batch in tqdm(enumerate(dl_test), total=len(dl_test)):
        imgs, labels, centers, descriptions = batch
        for img, label, center, description in zip(imgs, labels, centers, descriptions):
            start_token, text_tokens, vq_tokens, end_token = tokenizer.tokenize(img.to(device), center, text=description, return_np_uint16=True)
            test_vq_tokens.append(vq_tokens)
            test_text_tokens.append(text_tokens)
        if i % 2000 == 0 and i > 0:
            np.save(os.path.join(OUT_DIR, f"test_vq_tokens_{i*BATCH_SIZE + test_checkpoint}.npy"), np.concatenate(test_vq_tokens))
            np.save(os.path.join(OUT_DIR, f"test_text_tokens_{i*BATCH_SIZE + test_checkpoint}.npy"), np.concatenate(test_text_tokens))
            test_vq_tokens=[]
            test_text_tokens=[]
    if len(test_vq_tokens) > 0 and len(test_text_tokens) > 0:
        test_vq_tokens = np.concatenate(test_vq_tokens)
        test_text_tokens = np.concatenate(test_text_tokens)
        np.save(os.path.join(OUT_DIR, "test_vq_tokens_last.npy"), test_vq_tokens)
        np.save(os.path.join(OUT_DIR, "test_text_tokens_last.npy"), test_text_tokens)

if __name__ == '__main__':
    # print(get_latest_data_checkpoint("/scratch2/moritz_data/glyphazzn"))
    main()