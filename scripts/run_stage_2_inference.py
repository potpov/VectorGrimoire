from models import VQ_Transformer, Vector_VQVAE
from tokenizer import VQTokenizer
from dataset import VQDataset
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from svgwrite import Drawing

def tokens_to_drawing(tokens: Tensor, tokenizer: VQTokenizer, padded_individual_max_length = 9.5, stroke_width = 0.7) -> Drawing:
    return tokenizer.assemble_svg(*tokenizer.decode(tokens), padded_individual_max_length=padded_individual_max_length, stroke_width=stroke_width)


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # Load VQ Tranformer
    ckpt = "/scratch2/moritz_logs/VQ_Transformer/glyphazzn_B_simplified_single_code/checkpoints/epoch=30-step=2976.ckpt"
    state_dict = torch.load(ckpt, map_location=device)["state_dict"]
    state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    vq_transformer = VQ_Transformer(num_tokens=17387, max_seq_len=256, dim=512, depth=12, heads=8)
    vq_transformer.load_state_dict(state_dict)
    vq_transformer = vq_transformer.eval()

    # Load the VQVAE
    vq_model = Vector_VQVAE(codebook_size=128, image_loss="mse", vq_method="fsq")
    ckpt = "/scratch2/moritz_logs/SVG_VQVAE/Stage1/glyphazzn_B_simplified_single_code/checkpoints/last-v3.ckpt"
    state_dict = torch.load(ckpt, map_location=device)["state_dict"]
    # map keys to remove the prefix "model."
    state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    vq_model.load_state_dict(state_dict)
    vq_model = vq_model.eval()

    # make the Tokenizer
    tokenizer = VQTokenizer(vq_model=vq_model,
                            full_image_res=128,
                            context_length=256,
                            tokens_per_patch=1)
    
    ds = VQDataset("/scratch2/moritz_data/glyphazzn/B_simplified/tokenized/split.csv",
               context_length=256,
               train = False)
    
    for idx in range(0, 20):
        context = ds.__getitem__(idx)[0][:range(4, 125, 2)[idx]]
        gt_input = tokenizer.decode(context, ignore_eos=True)
        tokenizer.assemble_svg(gt_input[0], gt_input[1], 9.5, 0.7).saveas(f"output/val_{idx}_input.svg")
        if context.dim() == 1:
            context = context.unsqueeze(0)

        out, reason = vq_transformer.generate_with_constraint(context,
                                                            tokenizer.special_token_mapping.get("<PAD>"), 
                                                            tokenizer.special_token_mapping.get("<EOS>"), 
                                                            patch_idx_range=(tokenizer.start_of_patch_token_idx, tokenizer.start_of_pos_token_idx),
                                                            pos_idx_range=(tokenizer.start_of_pos_token_idx, tokenizer.num_tokens))
        out = out[0]
        if out[-1] == tokenizer.special_token_mapping.get("<EOS>"):
            tokens_to_drawing(out, tokenizer).saveas(f"output/val_{idx}_generated.svg")