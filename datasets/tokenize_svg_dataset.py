from thesis.tokenizer import VQTokenizer
from thesis.dataset import CenterShapeLayersFromSVGDataset
from thesis.models import Vector_VQVAE
import numpy as np
import torch
from glob import glob
import os
from tqdm import tqdm

def main():
    CONTEXT_LENGTH = 256
    MODEL_WEIGHTS_PATH = "/scratch2/moritz_logs/SVG_VQVAE/Stage1/glyphazzn_B_simplified_single_code/checkpoints/last-v3.ckpt"
    OUT_DIR = "/scratch2/moritz_data/glyphazzn/B_simplified/tokenized"
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    # load path to all svg files
    csv_path = "/scratch2/moritz_data/glyphazzn/B_simplified/split.csv"
    print("Loading dataset..")
    ds_train = CenterShapeLayersFromSVGDataset(csv_path, 3, 128, train=True, individual_max_length=7.5, stroke_width=0.7, max_shapes_per_svg=CONTEXT_LENGTH//2)
    ds_test = CenterShapeLayersFromSVGDataset(csv_path, 3, 128, train=False, individual_max_length=7.5, stroke_width=0.7, max_shapes_per_svg=CONTEXT_LENGTH//2)
    print("Loading model..")
    model = Vector_VQVAE(vq_method="fsq")
    state_dict = torch.load(MODEL_WEIGHTS_PATH)["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except:
        model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
    model = model.eval()
    tokenizer = VQTokenizer(model, 128, CONTEXT_LENGTH, 1)
    print("Number of Tokens: ",tokenizer.num_tokens)

    train_tokens = []
    print("Processing training set..")
    for i in tqdm(range(len(ds_train))):
        imgs, _, centers = ds_train[i]
        tokens = tokenizer.tokenize(imgs, centers, return_np_uint16=True)
        train_tokens.append(tokens)
        if i % 100 == 0:
            np.save(os.path.join(OUT_DIR, "train_tokens.npy"), np.concatenate(train_tokens))
    train_tokens = np.concatenate(train_tokens)
    np.save(os.path.join(OUT_DIR, "train_tokens.npy"), train_tokens)

    test_tokens = []
    print("Processing test set..")
    for i in tqdm(range(len(ds_test))):
        imgs, _, centers = ds_test[i]
        tokens = tokenizer.tokenize(imgs, centers, return_np_uint16=True)
        test_tokens.append(tokens)
        if i % 100 == 0:
            np.save(os.path.join(OUT_DIR, "test_tokens.npy"), np.concatenate(test_tokens))
    test_tokens = np.concatenate(test_tokens)
    np.save(os.path.join(OUT_DIR, "test_tokens.npy"), test_tokens)

if __name__ == '__main__':
    main()