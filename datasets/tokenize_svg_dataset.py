from thesis.tokenizer import VQTokenizer
from thesis.dataset import GlyphazznStage1Dataset
from thesis.models import Vector_VQVAE
import numpy as np
import torch
from glob import glob
import os
from tqdm import tqdm
import yaml
import pandas as pd

def main():
    CONTEXT_LENGTH = 256
    MODEL_WEIGHTS_PATH = "/scratch2/moritz_logs/SVG_VQVAE/Stage1/glyphazzn_full_single_code/checkpoints/last.ckpt"
    OUT_DIR = "/scratch2/moritz_data/glyphazzn/tokenized"
    TOP_LEVEL_DIR = "/scratch2/moritz_data/glyphazzn/svgs_simplified"
    CONFIG_PATH = "/home/mfeuerpfeil/master/thesis/configs/SVG_VQVAE_Stage1.yaml"

    with open(CONFIG_PATH, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config['data_params']["max_shapes_per_svg"] = CONTEXT_LENGTH//2  # because each shape also needs a position token
        
    csv_path = os.path.join(TOP_LEVEL_DIR, "split.csv")
    df = pd.read_csv(csv_path)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    
    
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    train_paths = train_df["path"].values
    test_paths = test_df["path"].values

    print("Loading model..")
    model = Vector_VQVAE(**config['model_params'])
    state_dict = torch.load(MODEL_WEIGHTS_PATH)["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except:
        model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
    model = model.eval()
    tokenizer = VQTokenizer(model, config["data_params"]["width"], 1, "bert-base-uncased")

    print("Loading dataset..")
    ds_train = GlyphazznStage1Dataset(train=True, **config['data_params'])
    ds_test = GlyphazznStage1Dataset(train=False, **config['data_params'])
    print("Number of Tokens: ",tokenizer.num_tokens)

    train_vq_tokens = []
    train_text_tokens = []
    print("Processing training set..")
    for i in tqdm(range(len(ds_train))):
        imgs, label, centers, description = ds_train._get_full_item(i)
        start_token, text_tokens, vq_tokens, end_token = tokenizer.tokenize(imgs, centers, text=description, return_np_uint16=True)
        train_vq_tokens.append(vq_tokens)
        train_text_tokens.append(text_tokens)
        if i % 100 == 0:
            np.save(os.path.join(OUT_DIR, "train_vq_tokens.npy"), np.concatenate(train_vq_tokens))
            np.save(os.path.join(OUT_DIR, "train_text_tokens.npy"), np.concatenate(train_text_tokens))
    train_vq_tokens = np.concatenate(train_vq_tokens)
    train_text_tokens = np.concatenate(train_text_tokens)
    np.save(os.path.join(OUT_DIR, "train_vq_tokens.npy"), train_vq_tokens)
    np.save(os.path.join(OUT_DIR, "train_text_tokens.npy"), train_text_tokens)

    test_vq_tokens = []
    test_text_tokens = []
    print("Processing test set..")
    for i in tqdm(range(len(ds_test))):
        imgs, label, centers, description = ds_test._get_full_item(i)
        start_token, text_tokens, vq_tokens, end_token = tokenizer.tokenize(imgs, centers, text=description, return_np_uint16=True)
        test_vq_tokens.append(vq_tokens)
        test_text_tokens.append(text_tokens)
        if i % 100 == 0:
            np.save(os.path.join(OUT_DIR, "test_vq_tokens.npy"), np.concatenate(test_vq_tokens))
            np.save(os.path.join(OUT_DIR, "test_text_tokens.npy"), np.concatenate(test_text_tokens))
    test_vq_tokens = np.concatenate(test_vq_tokens)
    test_text_tokens = np.concatenate(test_text_tokens)
    np.save(os.path.join(OUT_DIR, "test_vq_tokens.npy"), test_vq_tokens)
    np.save(os.path.join(OUT_DIR, "test_text_tokens.npy"), test_text_tokens)

if __name__ == '__main__':
    main()