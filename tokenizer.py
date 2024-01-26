from typing import Tuple

import numpy as np
from models import Vector_VQVAE
import torch
from torch import Tensor

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
    
    def decode_patches(self, tokens: Tensor) -> Tensor:
        """
        Decodes the patches from the tokens.

        Args:
            tokens (Tensor): Tensor of shape (num_patches, self.tokens_per_patch)

        Returns:
            Tensor: Tensor of shape (num_patches, channels, patch_res, patch_res)
        """
        with torch.no_grad():
            patches = self.vq_model.decode_from_indices(tokens - self.start_of_patch_token_idx)
        return patches
    
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