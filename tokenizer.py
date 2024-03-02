from typing import Iterable, List, Tuple, Union
from utils import calculate_global_positions, shapes_to_drawing, drawing_to_tensor

import numpy as np
from models.svg_vqvae import Vector_VQVAE
import torch
from torch import Tensor
from svgwrite import Drawing
from transformers import BertTokenizer, BertModel,PreTrainedTokenizerBase
from torch import nn
from svg_fixing import get_fixed_svg_render 

class VQTokenizer(nn.Module):
    """
    Tokenizer for the SVG-VQVAE model. It tokenizes the patches of the rasterized SVGs and their middle positions + some special tokens + text conditioning.

    Args:
        - vq_model (Vector_VQVAE): VQVAE model to use for patch tokenization
        - full_image_res (int): Full resolution of the rasterized SVGs
        - tokens_per_patch (int): Number of tokens per patch
        - text_encoder_str (str): huggingface string of the BERT text encoder to use, default: bert-base-uncased
        - device (str, optional): Device to use. Defaults to "cpu".
        - use_text_encoder_only (bool, optional): Whether to use the text encoder only. Defaults to False. Used to bnenefit from special token mapping and text tokenization without the need for a VQVAE model.
    """

    def __init__(self, vq_model: Vector_VQVAE, 
                 full_image_res: int, 
                 tokens_per_patch:int, 
                 text_encoder_str: str = "bert-base-uncased", 
                 device = "cpu",
                 use_text_encoder_only: bool = False,
                 codebook_size:int = None,
                 **kwargs) -> None:

        super(VQTokenizer, self).__init__()
        self.text_encoder_str = text_encoder_str
        self.full_image_res = full_image_res
        self.tokens_per_patch = tokens_per_patch
        self.max_num_pos_tokens = self.full_image_res ** 2  # for now this is just resolution squared, could be quantized to a smaller number of positions later
        self.device = device
        self.use_text_encoder_only = use_text_encoder_only
        if self.use_text_encoder_only:
            self.vq_model = None
            self.codebook_size = codebook_size
        else:
            self.vq_model = vq_model.to(device)
            self.codebook_size = self.vq_model.codebook_size
        
        self.text_tokenizer: PreTrainedTokenizerBase = BertTokenizer.from_pretrained(self.text_encoder_str)
        assert self.text_tokenizer.vocab_size < 65535, "VQTokenizer only supports 16-bit np.ushort encoded tokens, but the text tokenizer exceeds that."

        # CLS and SEP are handled by the text embedding model
        self.special_token_mapping = {
            "<SOS>": 0,  # start of sequence
            "<BOS>": 1,  # beginning of SVG, separates text tokens from SVG
            "<EOS>": 2,  # end of sequence
            "<PAD>": 3,  # padding
        }

        self.start_of_patch_token_idx = len(self.special_token_mapping)
        self.start_of_pos_token_idx = self.start_of_patch_token_idx + self.codebook_size * self.tokens_per_patch  # TODO validate (everywhere) if stuff needs a +1
        self.num_tokens = self.start_of_pos_token_idx + self.max_num_pos_tokens
        
    def _is_position(self, token: int) -> bool:
        return token >= self.start_of_pos_token_idx and token <= self.num_tokens

    def _is_patch(self, token: int) -> bool:
        return token >= self.start_of_patch_token_idx and token < self.start_of_pos_token_idx
    
    def _get_patch_idx_range(self) -> Tuple[int, int]:
        return self.start_of_patch_token_idx, self.start_of_pos_token_idx
    
    def _get_pos_idx_range(self) -> Tuple[int, int]:
        return self.start_of_pos_token_idx, self.num_tokens
    
    def tokenize_patches(self, patches: Tensor) -> Tensor:
        """
        Tokenizes the patches of the rasterized SVGs.

        Args:
            patches (Tensor): Tensor of shape (num_patches, channels, patch_res, patch_res)

        Returns:
            Tensor: Tensor of shape (num_patches, self.tokens_per_patch)
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Tokenizing patches is not supported when using the text encoder only.")
        with torch.no_grad():
            _, indices = self.vq_model.encode(patches, quantize=True)
        indices = indices.flatten().to(self.device)
        return indices + self.start_of_patch_token_idx
    
    def tokenize_positions(self, positions: Tensor) -> Tensor:
        """
        Tokenizes the positions of the patches of the rasterized SVGs.

        Args:
            positions (Tensor): Tensor of shape (num_pos, 2)

        Returns:
            Tensor: Tensor of shape (num_pos, 1)
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Tokenizing positions is not supported when using the text encoder only.")
        assert positions.mean() > 1., f"Positions should be scaled with the full image resolution already, got mean: {positions.mean()}"
        positions = positions[:, 0].round() + self.full_image_res * positions[:, 1].round()
        return positions + self.start_of_pos_token_idx
    
    def tokenize_text(self, text: str) -> Tensor:
        """
        Tokenizes the conditional text.

        Args:
            text (str): Text to tokenize

        Returns:
            Tensor: Tensor of shape (num_tokens) without any padding but with special tokens [CLS] and [SEP]
        """
        tokens = torch.tensor(self.text_tokenizer.encode(text, add_special_tokens=True), device = self.device)
        return tokens

    def forward(self):
        pass
    
    def tokenize(self, patches: Tensor, positions: Tensor, text:str, return_np_uint16:bool = False) -> Union[Tensor, Tensor] | Union[np.ndarray, np.ndarray]:
        """
        Tokenizes the patches and positions of the rasterized SVGs. Padding is done in the dataloader dynamically to avoid requiring a fixed context length during pre-tokenization.

        Args:
            - patches (Tensor): Tensor of shape (num_patches, channels, patch_res, patch_res)
            - positions (Tensor): Tensor of shape (num_pos, 2)
            - text (str): conditional text
            - return_np_uint16 (bool, optional): Whether to return the tokens as np.uint16. Defaults to False.
            - batched (bool, optional): Whether the input is batched or not.

        Returns:
            - start_token: [<SOS>], either Tensor or np.ndarray
            - text_tokens: [<CLS>, ...text..., <SEP>], no padding, CLS and SEP come from text tokenizer, either Tensor or np.ndarray
            - vq_tokens: [<BOS>, patch_tokens, pos_token, patch_tokens, pos_token, ...], no padding, either Tensor or np.ndarray
            - end_token: [<EOS>], either Tensor or np.ndarray
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Tokenizing patches/positions is not supported when using the text encoder only.")
        patch_tokens = self.tokenize_patches(patches).cpu()
        pos_tokens = self.tokenize_positions(positions)
        text_tokens = self.tokenize_text(text)
        if self.tokens_per_patch == 1:
            vq_tokens = torch.stack([patch_tokens, pos_tokens], dim=1).reshape(-1).int()
        else:
            raise NotImplementedError("Merging not implemented for tokens_per_patch > 1")
        
        # NOTE: this is now done manually in the tokenization script as <SOS> needs to be put before the text tokens but I want to keep text and SVG tokens separate
        start_token = (self.special_token_mapping["<SOS>"]) * torch.ones(1).int()
        end_token = (self.special_token_mapping["<EOS>"]) * torch.ones(1).int()
        bos_token = (self.special_token_mapping["<BOS>"]) * torch.ones(1).int()

        vq_tokens = torch.cat([bos_token, vq_tokens], dim=0)

        # final_tokens = torch.cat([start_token, vq_tokens, end_token], dim=0)

        if return_np_uint16:
            vq_tokens = vq_tokens.numpy().astype(np.ushort)
            text_tokens = text_tokens.numpy().astype(np.ushort)
            start_token = start_token.numpy().astype(np.ushort)
            end_token = end_token.numpy().astype(np.ushort)
        
        return start_token, text_tokens, vq_tokens, end_token
    
    def decode_patches(self, tokens: Tensor, raster:bool = False) -> Tensor:
        """
        Decodes the patches from the tokens into bezier points.

        Args:
            tokens (Tensor): Tensor of shape (num_patches, self.tokens_per_patch)
            raster (bool, optional): Whether to return the rasterized patches. Defaults to False.

        Returns:
            Tensor: Tensor of shape (num_patches, channels, patch_res, patch_res)
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Decoding patches is not supported when using the text encoder only.")
        with torch.no_grad():
            out, _ = self.vq_model.decode_from_indices(tokens - self.start_of_patch_token_idx)
        if raster:
            return out[0]
        else:
            return out[2]
    
    def decode_positions(self, tokens: Tensor) -> Tensor:
        """
        Decodes the positions from the tokens.

        Args:
            tokens (Tensor): Tensor of shape (num_pos, 1)

        Returns:
            Tensor: Tensor of shape (num_pos, 2)
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Decoding positions is not supported when using the text encoder only.")
        tokens = tokens - self.start_of_pos_token_idx
        positions = torch.stack([tokens % self.full_image_res, tokens // self.full_image_res], dim=1)
        return positions
    
    def decode_text(self, tokens: Tensor) -> str:
        """
        Decodes the text from the tokens.

        Args:
            tokens (Tensor): Tensor of shape (num_tokens)

        Returns:
            str: Decoded text
        """
        text = self.text_tokenizer.decode(tokens, skip_special_tokens=True)
        return text
    
    def decode(self, tokens: Tensor, ignore_special_tokens: bool = False):
        """
        Decodes the patches and positions from the tokens.

        Args:
            tokens (Tensor): Tensor of shape (num_tokens)
            ignore_special_tokens (bool, optional): Whether to ignore the required special tokens like BOS and EOS. Defaults to False.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of tensors of shape (num_patches, channels, patch_res, patch_res) and (num_pos, 2)
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Decoding patches/positions is not supported when using the text encoder only.")
        # remove all occurence of <PAD> token
        tokens = tokens[tokens != self.special_token_mapping["<PAD>"]]

        assert tokens.ndim == 1, f"Tokens should be 1D, got shape {tokens.shape}"
        assert tokens.size(0) > 3, f"Tokens should have at least 4 elements, got {tokens.size(0)}"
        if not ignore_special_tokens:
            assert tokens[0] == self.special_token_mapping["<BOS>"], f"First token should be <BOS>, got {tokens[0]}"
            assert tokens[-1] == self.special_token_mapping["<EOS>"] or tokens[-1] == self.special_token_mapping["<PAD>"], f"Last token should be <EOS> or <PAD>, got {tokens[-1]}"
        if tokens[-1] == self.special_token_mapping["<EOS>"]:
            tokens = tokens[:-1]
        if tokens[0] == self.special_token_mapping["<BOS>"]:
            tokens = tokens[1:]
        if self._is_patch(tokens[-1]):
            # print("[INFO] Last token is a patch token, removing it.")
            tokens = tokens[:-1]
        if self.tokens_per_patch == 1:
            assert tokens.size(0) % 2 == 0, f"Number of tokens should be even, got {tokens.size(0)}"
        patch_tokens = tokens[::2]
        pos_tokens = tokens[1::2]
        bezier_points = self.decode_patches(patch_tokens)
        positions = self.decode_positions(pos_tokens)
        return bezier_points, positions
    
    def assemble_svg(self, bezier_points: Tensor, center_positions: Tensor, padded_individual_max_length: float, stroke_width: float, w=128., num_strokes_to_paint:int = 0) -> Drawing:
        global_shapes = calculate_global_positions(bezier_points, padded_individual_max_length, center_positions)[:,0]
        reconstructed_drawing = shapes_to_drawing(global_shapes, stroke_width=stroke_width, w=w, num_strokes_to_paint=num_strokes_to_paint)
        return reconstructed_drawing
    
    def _tokens_to_image_tensor(self, tokens:Tensor, post_process:bool = True, num_strokes_to_paint: int = 0) -> Tensor:
        bezier_points, positions = self.decode(tokens, ignore_special_tokens=True)
        if post_process:
            return_tensor = get_fixed_svg_render(bezier_points, positions, "min_dist_clip", 0.7, 9.5, 480, 4.5, num_strokes_to_paint=num_strokes_to_paint)
        else:
            drawing = self.assemble_svg(bezier_points, positions, 9.5, 0.7, w=480, num_strokes_to_paint=num_strokes_to_paint)
            return_tensor = drawing_to_tensor(drawing)
        return return_tensor