from typing import Tuple
from thesis.utils import calculate_global_positions, shapes_to_drawing

import numpy as np
from models import Vector_VQVAE
import torch
from torch import Tensor
from svgwrite import Drawing

class VQTokenizer:
    """
    Tokenizer for the SVG-VQVAE model. It tokenizes the patches of the rasterized SVGs and their middle positions + some special tokens.

    """

    def __init__(self, vq_model: Vector_VQVAE, full_image_res: int, context_length : int, tokens_per_patch:int) -> None:
        self.vq_model = vq_model
        self.full_image_res = full_image_res
        self.codebook_size = self.vq_model.codebook_size
        self.tokens_per_patch = tokens_per_patch
        self.max_num_pos_tokens = self.full_image_res ** 2  # for now this is just resolution squared, could be quantized to a smaller number of positions later
        self.context_length = context_length

        self.special_token_mapping = {
            "<SOS>": 0,
            "<EOS>": 1,
            "<PAD>": 2,
        }

        self.start_of_patch_token_idx = len(self.special_token_mapping)
        self.start_of_pos_token_idx = self.start_of_patch_token_idx + self.codebook_size * self.tokens_per_patch  # TODO validate if this needs a +1
        self.num_tokens = self.start_of_pos_token_idx + self.max_num_pos_tokens
        

    def tokenize_patches(self, patches: Tensor) -> Tensor:
        """
        Tokenizes the patches of the rasterized SVGs.

        Args:
            patches (Tensor): Tensor of shape (num_patches, channels, patch_res, patch_res)

        Returns:
            Tensor: Tensor of shape (num_patches, self.tokens_per_patch)
        """
        with torch.no_grad():
            _, indices = self.vq_model.encode(patches, quantize=True)
        indices = indices.flatten()
        return indices + self.start_of_patch_token_idx
    
    def tokenize_positions(self, positions: Tensor) -> Tensor:
        """
        Tokenizes the positions of the patches of the rasterized SVGs.

        Args:
            positions (Tensor): Tensor of shape (num_pos, 2)

        Returns:
            Tensor: Tensor of shape (num_pos, 1)
        """
        assert positions.mean() > 1., f"Positions should be scaled with the full image resolution already, got mean: {positions.mean()}"
        positions = positions[:, 0].round() + self.full_image_res * positions[:, 1].round()
        return positions + self.start_of_pos_token_idx
        
    def tokenize(self, patches: Tensor, positions: Tensor, return_np_uint16:bool = False) -> Tensor | np.ndarray:
        """
        Tokenizes the patches and positions of the rasterized SVGs.

        Args:
            patches (Tensor): Tensor of shape (num_patches, channels, patch_res, patch_res)
            positions (Tensor): Tensor of shape (num_pos, 2)

        Returns:
            Tensor: Tensor of shape (num_patches + num_pos, self.tokens_per_patch)
            or
            np.ndarray: Numpy array of shape (num_patches + num_pos, self.tokens_per_patch) with dtype np.ushort
        """
        patch_tokens = self.tokenize_patches(patches)
        pos_tokens = self.tokenize_positions(positions)
        if self.tokens_per_patch == 1:
            alternating_tokens = torch.stack([patch_tokens, pos_tokens], dim=1).reshape(-1, 1).int()
        else:
            raise NotImplementedError("Merging not implemented for tokens_per_patch > 1")
        
        start_token = (self.special_token_mapping["<SOS>"]) * torch.ones(1, 1).int()
        end_token = (self.special_token_mapping["<EOS>"]) * torch.ones(1, 1).int()

        final_tokens = torch.cat([start_token, alternating_tokens, end_token], dim=0)
        # if final_tokens.size(0) < self.context_length:
        #     final_tokens = torch.cat([final_tokens, self.special_token_mapping["<PAD>"] * torch.ones(self.context_length - final_tokens.size(0))], dim=0)
        # else:
        #     final_tokens = final_tokens[:self.context_length]
        if return_np_uint16:
            final_tokens = final_tokens.numpy().astype(np.ushort)
        
        return final_tokens
    
    def decode_patches(self, tokens: Tensor, raster:bool = False) -> Tensor:
        """
        Decodes the patches from the tokens into bezier points.

        Args:
            tokens (Tensor): Tensor of shape (num_patches, self.tokens_per_patch)
            raster (bool, optional): Whether to return the rasterized patches. Defaults to False.

        Returns:
            Tensor: Tensor of shape (num_patches, channels, patch_res, patch_res)
        """
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
        tokens = tokens - self.start_of_pos_token_idx
        positions = torch.stack([tokens % self.full_image_res, tokens // self.full_image_res], dim=1)
        return positions
    
    def decode(self, tokens: Tensor, ignore_eos: bool = False):
        """
        Decodes the patches and positions from the tokens.

        Args:
            tokens (Tensor): Tensor of shape (num_tokens)

        Returns:
            Tuple[Tensor, Tensor]: Tuple of tensors of shape (num_patches, channels, patch_res, patch_res) and (num_pos, 2)
        """
        # remove all occurence of <PAD> token
        tokens = tokens[tokens != self.special_token_mapping["<PAD>"]]

        assert tokens.ndim == 1, f"Tokens should be 1D, got shape {tokens.shape}"
        assert tokens.size(0) > 3, f"Tokens should have at least 4 elements, got {tokens.size(0)}"
        assert tokens[0] == self.special_token_mapping["<SOS>"], f"First token should be <SOS>, got {tokens[0]}"
        if not ignore_eos:
            assert tokens[-1] == self.special_token_mapping["<EOS>"], f"Last token should be <EOS>, got {tokens[-1]}"
        tokens = tokens[1:-1]
        if self.tokens_per_patch == 1:
            assert tokens.size(0) % 2 == 0, f"Number of tokens should be even, got {tokens.size(0)}"
        patch_tokens = tokens[::2]
        pos_tokens = tokens[1::2]
        bezier_points = self.decode_patches(patch_tokens)
        positions = self.decode_positions(pos_tokens)
        return bezier_points, positions
    
    def assemble_svg(self, bezier_points: Tensor, center_positions: Tensor, padded_individual_max_length: float, stroke_width: float) -> Drawing:
        global_shapes = calculate_global_positions(bezier_points, padded_individual_max_length, center_positions)[:,0]
        reconstructed_drawing = shapes_to_drawing(global_shapes, stroke_width=stroke_width, w=72.)
        return reconstructed_drawing