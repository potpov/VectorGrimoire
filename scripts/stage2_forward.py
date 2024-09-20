import os
import torch
import yaml
import numpy as np
from dataset import VQDataset
from models import VQ_SVG_Stage2,VSQ
from tokenizer import VQTokenizer

path = "/scratch2/moritz_data/glyphazzn/tokenized/full_training_text_tokens.npy"
path2 = "/scratch2/moritz_data/glyphazzn/tokenized/full_training_vq_tokens.npy"

text_tokens = np.load(path, allow_pickle=True)
vq_tokens = np.load(path2, allow_pickle=True)


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

with open(CONFIG_PATH, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
config['data_params']["max_shapes_per_svg"] = 512  # more than context length of 1024 (at least one patch and one pos token per svg shape) we'll never do I think
    
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

print("Loading model..")

model = VSQ(**config['model_params']).to(device)
state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)["state_dict"]
try:
    model.load_state_dict(state_dict)
except:
    model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
model = model.eval()
tokenizer = VQTokenizer(model, config["data_params"]["width"], 1, "bert-base-uncased")

train_ds = VQDataset("/scratch2/moritz_data/glyphazzn/tokenized/split_debug.csv", context_length=256, train=True)
test_ds = VQDataset("/scratch2/moritz_data/glyphazzn/tokenized/split_debug.csv", context_length=256, train=False)

stage2 = VQ_SVG_Stage2(tokenizer = tokenizer,
                       max_seq_len=256,
                       dim=512,
                       depth = 6,
                       heads = 4,
                       use_alibi_positional_bias= False).to(device)

# ----------------------------------
text_tokens, attention_mask, vq_tokens, vq_targets = train_ds[0]
text_tokens = text_tokens.unsqueeze(0).to(device)
attention_mask = attention_mask.unsqueeze(0).to(device)
vq_tokens = vq_tokens.unsqueeze(0).to(device)

# out = stage2.text_embedder.forward(text_tokens, attention_mask=attention_mask).last_hidden_state
stage2.forward(text_tokens, attention_mask, vq_tokens)
# ----------------------------------
input("End of script.. Press Enter to continue...")
