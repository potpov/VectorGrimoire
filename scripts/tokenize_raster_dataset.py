from torch.utils.data import DataLoader
from pathlib import Path
from models import VSQ
from tokenizer import RasterVQTokenizer
import numpy as np
import torch
import cv2
import yaml
import os
import resource
import torch.nn.functional as F
from tqdm import tqdm
from dataset import TiledMNIST, MNISTDataset
from utils import get_filter_function
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2

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


def tokenize_MNIST_augmented():

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    ######
    # setting 2 - 8x8 grid, black and white VSQ
    MODEL_WEIGHTS_PATH = "/raid/marco.cipriano/results/svg/Grimoire/VSQ/VSQ_MNIST_BW_P128_T14_P20_TH0.2/checkpoints/last.ckpt"
    OUT_PATH = "/raid/marco.cipriano/data/SVG/Grimoire/MNIST/mnist_tokenized/VSQ_MNIST_BW_P128_T14_P20_TH0.2AUG"
    CONFIG_PATH = "../configs/MNIST/MNIST_VSQ_BW.yaml"
    FILTER_TH = 0.2

    MNIST_SETTING = {
        "data_path": "/raid/marco.cipriano/data/SVG/Grimoire/MNIST/mnist_png",
        "train_batch_size": 1,
        "val_batch_size": 1,
        "patch_size": 128,
        "num_tiles_per_row": 14,
        "num_workers": 0,
        "pin_memory": False,
        "random_colors": False,
        "use_palette": False,
        "padding_frac": 0.1,
        "return_filename": True,
    }

    with open(CONFIG_PATH, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
    #################
    ###  MODEL
    print("Loading model..")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = VSQ(patch_size=MNIST_SETTING["patch_size"], **config['model_params'])
    if MODEL_WEIGHTS_PATH is not None:
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)["state_dict"]
        try:
            model.load_state_dict(state_dict)
        except:
            model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})

    filter_fn = None
    if FILTER_TH is not None:
        print("Filtering patches with less than ", FILTER_TH, " non-white pixels")
        filter_fn = get_filter_function(FILTER_TH, parse_patches=False)

    model = model.eval()
    model = model.to(device)
    tokenizer = RasterVQTokenizer(
        model,
        tokens_per_patch=1,
        do_tokenize_positions=False,
        patch_size=MNIST_SETTING["patch_size"],
        num_tiles_per_row=MNIST_SETTING["num_tiles_per_row"],
        device=device,
        filter_fn=filter_fn
    )
    print("Loading dataset..")
    datamodule = MNISTDataset(**MNIST_SETTING)
    datamodule.setup()

    dl_train = datamodule.train_dataloader()
    dl_test = datamodule.test_dataloader()

    all_paths = dl_train.dataset.image_paths
    all_desc = dl_train.dataset.labels

    print("Number of Tokens: ", tokenizer.num_tokens)
    patch_size = MNIST_SETTING["patch_size"]
    num_tiles_per_row = MNIST_SETTING["num_tiles_per_row"]
    padding_frac = MNIST_SETTING["padding_frac"]
    total_padding = total_padding = (int(patch_size * padding_frac) // 2) * 2
    new_dimension = (patch_size - total_padding) * num_tiles_per_row
    single_side_padding = total_padding // 2
    base_transforms = transforms.Compose(
        [
            transforms.Resize(new_dimension, antialias=True),
            transforms.RandomInvert(1.0),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    def patch(image):
        patches = []
        for i in range(0, image.shape[1], patch_size - single_side_padding * 2):
            for j in range(0, image.shape[2], patch_size - single_side_padding * 2):
                patch = image[:, i: i + patch_size - total_padding, j: j + patch_size - total_padding]
                patch = F.pad(patch,
                              (single_side_padding, single_side_padding, single_side_padding, single_side_padding),
                              value=1.)
                patches.append(patch)
        return patches

    num_shift, num_rotation, num_zoom, num_shear = 5, 5, 5, 1
    print("FINAL EXPECTED NUMBER OF SAMPLES: ", len(all_paths) * (num_shift + num_rotation + num_zoom + num_shear))
    shift_vals = np.random.uniform(-100, 100, (num_shift, 2))
    rotation_vals = np.random.uniform(-30, 30, num_rotation)
    zoom_vals = np.random.uniform(0.5, 1.5, num_zoom)
    shear_vals = np.random.uniform(-20, 20, num_shear)


    save_csv = {
        "index_in_numpy_array": [], "filename": [],
        "split": [], "label": [], "text_token_length": [],
        "vq_token_length": [], "description": []
    }
    vsq_token_array = []
    text_token_array = []
    full_token_array = []
    numpy_counter = 0

    for idx in tqdm(range(len(all_paths)), total=len(all_paths)):

        path = all_paths[idx]
        desc = all_desc[idx]
        all_variations = []
        image = Image.open(path)
        image = base_transforms(image)
        image = torch.where(image > 0.6, 1., 0.)  # makes binary

        # ORIGINAL IMAGE
        all_variations.append(image)

        # ALL ROTATIONS
        for angle in rotation_vals:
            all_variations.append(v2.functional.affine(image, angle=angle, translate=(0, 0), fill=1., shear=0, scale=1))

        # ALL SHIFTING
        for a, b in shift_vals:
            all_variations.append(
                v2.functional.affine(image, angle=0, translate=(float(a), float(b)), fill=1., shear=0, scale=1))

            # # ALL ZOOMS
        for zoom_val in zoom_vals:
            all_variations.append(v2.functional.affine(image, angle=0, translate=(0, 0), fill=1., shear=0, scale=zoom_val))
        #
        # # ALL SHEARS
        for shear_val in shear_vals:
            all_variations.append(v2.functional.affine(image, angle=0, translate=(0, 0), fill=1., shear=shear_val, scale=1))

        # PROCESSING AND SAVING
        description = f"number {desc}"
        for variation in all_variations:
            variation = torch.stack(patch(variation))
            start_token, text_tokens, vq_tokens, end_token = tokenizer.tokenize(variation.cuda(), text=description,
                                                                                return_np_uint16=True)

            assert vq_tokens.max() <= tokenizer.num_tokens, f"out of boundary tokens in iteration ??. Max token: {vq_tokens.max()}"
            save_csv["index_in_numpy_array"].append(numpy_counter)
            save_csv["filename"].append("")
            save_csv["split"].append("train")
            save_csv["description"].append(description.lower())
            save_csv["label"].append(desc)
            save_csv["text_token_length"].append(len(text_tokens))
            save_csv["vq_token_length"].append(len(vq_tokens))
            vsq_token_array.append(vq_tokens)
            text_token_array.append(text_tokens)
            full_token_array.append(np.concatenate([start_token, text_tokens, vq_tokens, end_token]))
            numpy_counter += 1

    #####
    ## adding test split without augmentations
    for i, batch in tqdm(enumerate(dl_test), total=len(dl_test), desc=f"processing test"):
        imgs, labels, _, descriptions, filenames = batch
        imgs = imgs.to(device)
        imgs = torch.where(imgs > 0.6, 1., 0.)  # makes binary
        start_token, text_tokens, vq_tokens, end_token = tokenizer.tokenize(imgs, text=descriptions[0],
                                                                            return_np_uint16=True)

        assert vq_tokens.max() <= tokenizer.num_tokens, f"out of boundary tokens in iteration {i}. Max token: {vq_tokens.max()}"
        save_csv["index_in_numpy_array"].append(numpy_counter)
        save_csv["filename"].append(filenames[0])
        save_csv["split"].append("test")
        save_csv["description"].append(descriptions[0].lower())
        save_csv["label"].append(labels[0])
        save_csv["text_token_length"].append(len(text_tokens))
        save_csv["vq_token_length"].append(len(vq_tokens))
        vsq_token_array.append(vq_tokens)
        text_token_array.append(text_tokens)
        full_token_array.append(np.concatenate([start_token, text_tokens, vq_tokens, end_token]))
        numpy_counter += 1

    np.save(os.path.join(OUT_PATH, "vsq_tokenized.npy"), np.concatenate(vsq_token_array))
    np.save(os.path.join(OUT_PATH, "text_tokenized.npy"), np.concatenate(text_token_array))
    np.save(os.path.join(OUT_PATH, "full_tokenized.npy"), np.concatenate(full_token_array))
    df = pd.DataFrame(save_csv)
    df.to_csv(os.path.join(OUT_PATH, "split.csv"), index=False)


def tokenize_MNIST():

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    ######
    # setting 1 - 8x8 grid, full random color VSQ
    # MODEL_WEIGHTS_PATH = "/raid/marco.cipriano/results/svg/Grimoire/VSQ/TiledMNIST/checkpoints/last.ckpt"
    # CONFIG_PATH = "configs/MNIST_VSQ.yaml"
    # OUT_PATH = "/home/marco.cipriano/data/SVG/Grimoire/MNIST/8x8_randomcolor"

    ######
    # setting 2 - 8x8 grid, black and white VSQ
    # MODEL_WEIGHTS_PATH = "/raid/marco.cipriano/results/svg/Grimoire/VSQ/VSQ_MNIST_BW_P128_T14_P20_TH0.2/checkpoints/last.ckpt"
    MODEL_WEIGHTS_PATH = "/raid/marco.cipriano/results/svg/Grimoire/VSQ/VSQ_MNIST_BW_P128_T6_P20_TH0.1/checkpoints/last.ckpt"
    # OUT_PATH = "/raid/marco.cipriano/data/SVG/Grimoire/MNIST/mnist_tokenized/VSQ_MNIST_BW_P128_T14_P20_TH0.2"
    OUT_PATH = "/raid/marco.cipriano/data/SVG/Grimoire/MNIST/mnist_tokenized/VSQ_MNIST_BW_P128_T6_P20_TH0.2"
    CONFIG_PATH = "/home/marco.cipriano/projects/Grimoire/configs/MNIST/MNIST_VSQ_BW.yaml"
    FILTER_TH = 0.2

    MNIST_SETTING = {
        "data_path": "/raid/marco.cipriano/data/SVG/Grimoire/MNIST/mnist_png",
        "train_batch_size": 1,
        "val_batch_size": 1,
        "patch_size": 128,
        "num_tiles_per_row": 6,
        "num_workers": 0,
        "pin_memory": False,
        "random_colors": False,
        "use_palette": False,
        "padding_frac": 0.1,
        "return_filename": True,
    }

    with open(CONFIG_PATH, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
    # copy model weights and config to OUT_PATH
    if MODEL_WEIGHTS_PATH is not None and os.path.exists(MODEL_WEIGHTS_PATH):
        os.system(f"cp {MODEL_WEIGHTS_PATH} {OUT_PATH}")
    else:
        raise ValueError(f"Model weights not found at {MODEL_WEIGHTS_PATH}")
    if CONFIG_PATH is not None and os.path.exists(CONFIG_PATH):
        os.system(f"cp {CONFIG_PATH} {OUT_PATH}")
    else:
        raise ValueError(f"Config not found at {CONFIG_PATH}")

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
    model = VSQ(patch_size=MNIST_SETTING["patch_size"], **config['model_params'])
    if MODEL_WEIGHTS_PATH is not None:
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)["state_dict"]
        try:
            model.load_state_dict(state_dict)
        except:
            model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})

    filter_fn = None
    if FILTER_TH is not None:
        print("Filtering patches with less than ", FILTER_TH, " non-white pixels")
        filter_fn = get_filter_function(FILTER_TH, parse_patches=False)

    model = model.eval()
    model = model.to(device)
    tokenizer = RasterVQTokenizer(
        model,
        tokens_per_patch=1,
        do_tokenize_positions=False,
        patch_size=MNIST_SETTING["patch_size"],
        num_tiles_per_row=MNIST_SETTING["num_tiles_per_row"],
        device=device,
        filter_fn=filter_fn
    )

    print("Loading dataset..")
    datamodule = MNISTDataset(**MNIST_SETTING)
    datamodule.setup()

    dl_train = datamodule.train_dataloader()
    dl_test = datamodule.test_dataloader()

    print("Number of Tokens: ", tokenizer.num_tokens)

    print("Processing training set..")

    save_csv = {
        "index_in_numpy_array": [], "filename": [],
        "split": [], "label": [], "text_token_length": [],
        "vq_token_length": [], "description": []
    }
    vsq_token_array = []
    text_token_array = []
    full_token_array = []
    numpy_counter = 0

    for split_name, split in {"train": dl_train, "test": dl_test}.items():
        for i, batch in tqdm(enumerate(split), total=len(split), desc=f"processing {split_name}"):
            imgs, labels, _, descriptions, filenames = batch
            # print(imgs, labels, centers, descriptions, filenames)
            imgs = imgs.to(device)
            # imgs = torch.where(imgs > 0.6, 1., 0.)  # makes binary
            # print(imgs.shape, descriptions, filenames)
            start_token, text_tokens, vq_tokens, end_token = tokenizer.tokenize(
                imgs,
                text=descriptions[0],
                include_positions=True,
                return_np_uint16=True
            )
            # debug
            # rasterized_gt = tokenizer._tokens_to_image_tensor(
            #     torch.asarray(np.concatenate([vq_tokens, end_token])).int().cuda(),
            #     ignore_special_tokens=True,
            #     only_patch_tokens=False
            # )
            assert vq_tokens.max() <= tokenizer.num_tokens, f"out of boundary tokens in iteration {i}. Max token: {vq_tokens.max()}"
            save_csv["index_in_numpy_array"].append(numpy_counter)
            save_csv["filename"].append(filenames[0])
            save_csv["split"].append(split_name)
            save_csv["description"].append(descriptions[0].lower())
            save_csv["label"].append(labels[0])
            save_csv["text_token_length"].append(len(text_tokens))
            save_csv["vq_token_length"].append(len(vq_tokens))
            # print("vq_tokens: ",vq_tokens)
            vsq_token_array.append(vq_tokens)
            text_token_array.append(text_tokens)
            full_token_array.append(np.concatenate([start_token, text_tokens, vq_tokens, end_token]))
            numpy_counter += 1

    np.save(os.path.join(OUT_PATH, "vsq_tokenized.npy"), np.concatenate(vsq_token_array))
    np.save(os.path.join(OUT_PATH, "text_tokenized.npy"), np.concatenate(text_token_array))
    np.save(os.path.join(OUT_PATH, "full_tokenized.npy"), np.concatenate(full_token_array))
    df = pd.DataFrame(save_csv)
    df.to_csv(os.path.join(OUT_PATH, "split.csv"), index=False)

if __name__ == '__main__':
    # print(get_latest_data_checkpoint("/scratch2/moritz_data/glyphazzn"))
    # tokenize_MNIST_augmented()
    tokenize_MNIST()