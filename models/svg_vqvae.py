from typing import Tuple, Union
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import wandb
from utils import log_all_images, tensor_to_histogram_image, calculate_global_positions, shapes_to_drawing, svg_string_to_tensor
from torchvision.models import ResNet, resnet18
from models.vq_vae import VectorQuantizer
from models.mlp_vector_head import MLPVectorHeadFixed, CNNVectorHead
from models.mlp import MultiLayerPerceptron
from vector_quantize_pytorch import FSQ
from x_transformers import TransformerWrapper, Decoder
from transformers import BertModel
from svgwrite import Drawing
from einops import rearrange


class DeconvResNet(nn.Module):
    def __init__(self):
        super(DeconvResNet, self).__init__()

        # Define layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = F.sigmoid(self.deconv5(x))  # Using sigmoid for the final layer to scale values between 0 and 1

        return x, {}

class VQ_Transformer(nn.Module):
    """
    Stage 2 of the SVG-VQ-VAE pipeline.
    """

    def __init__(self,
                num_tokens: int = 20000,
                max_seq_len: int = 512,
                dim: int = 512,
                depth: int = 12,
                heads: int = 8,
                text_encoder_str: str = "bert-base-uncased",
                 **kwargs) -> None:
        
        super(VQ_Transformer, self).__init__()

        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.text_encoder_str = text_encoder_str
        self.text_encoder = BertModel.from_pretrained(self.text_encoder_str)

        self.model = TransformerWrapper(
            num_tokens = self.num_tokens,
            max_seq_len = self.max_seq_len,
            attn_layers = Decoder(
                dim = self.dim,
                depth = self.depth,
                heads = self.heads,
                attn_flash = True,
                alibi_pos_bias = True, # turns on ALiBi positional embedding
                alibi_num_heads = 4    # only use ALiBi for 4 out of the 8 heads, so other 4 heads can still attend far distances
            )
        )

    def forward(self, x: Tensor, **kwargs) -> Union[Tensor, dict]:
        return self.model.forward(x), {}
    
    def loss_function(self, targets: Tensor, pred_probabilities: Tensor, **kwargs) -> dict:
        loss = F.cross_entropy(pred_probabilities,targets)
        return {'loss': loss}
    
    def _generate(self, input_tokens: Tensor, eos_token: int) -> Tensor:
        """
        Generates tokens from the given input tokens but has no constraints on the order of the type of tokens. Can produce invalid SVGs.
        """
        with torch.no_grad():  # no need to track gradients when generating tokens
            while input_tokens.shape[1] < self.max_seq_len:
                predictions, _ = self.forward(input_tokens)
                # get the last predicted token
                last_token = predictions[:, -1:, :].argmax(dim=-1)
                # check if the last token is the <EOS> token
                # append the last token to the input tokens
                input_tokens = torch.cat([input_tokens, last_token], dim=1)
                if last_token.item() == eos_token:
                    reason = "EOS token reached"
                    break
                elif input_tokens.shape[1] >= self.max_seq_len:
                    reason = "Max sequence length reached"
                    break
        return input_tokens, reason
    
    def generate_with_constraint(self, input_tokens: Tensor, pad_token:int, eos_token: int, patch_idx_range: Tuple[int, int], pos_idx_range: Tuple[int, int]) -> Tensor:
        assert pos_idx_range[0] >= patch_idx_range[1], "pos_idx_range must start after patch_idx_range ends"
        if (input_tokens[:, -1] >= pos_idx_range[0]).all() and (input_tokens[:, -1] <= pos_idx_range[1]).all():
            required_token = "patch"
        elif (input_tokens[:, -1] >= patch_idx_range[0]).all() and (input_tokens[:, -1] <= patch_idx_range[1]).all():
            required_token = "pos"
        elif (input_tokens[:, -1] < patch_idx_range[0]).all():  # e.g. only <SOS> tokens in input
            required_token = "patch"
        else:
            raise ValueError("Last tokens in Input must be of the same type (special, patch, or pos).")

        with torch.no_grad():
            while input_tokens.shape[1] < self.max_seq_len:
                predictions, _ = self.forward(input_tokens)
                predictions[:, -1, pad_token] = -torch.inf  # mask the padding token
                if required_token == "patch":
                    predictions[:, -1, pos_idx_range[0]:pos_idx_range[1]] = -torch.inf
                    required_token = "pos"
                elif required_token == "pos":
                    predictions[:, -1, patch_idx_range[0]:patch_idx_range[1]] = -torch.inf
                    required_token = "patch"
                
                # get the last predicted token
                last_token = predictions[:, -1:, :].argmax(dim=-1)
                # check if the last token is the <EOS> token
                # append the last token to the input tokens
                input_tokens = torch.cat([input_tokens, last_token], dim=1)
                if last_token.item() == eos_token:
                    reason = "EOS token reached"
                    break
                elif input_tokens.shape[1] >= self.max_seq_len:
                    reason = "Max sequence length reached"
                    break
        return input_tokens, reason


class Vector_VQVAE(nn.Module):
    """
    Vector quantized pre-training of an autoencoder for SVG primitives.
    
    Input/Output are shape layers, no positions. Positions are decoded using the transformer in Stage II.

    TODO:
        - reduce encoding dimension from (512, 4, 4) to something more reasonable like (512, 2, 2), this must also be changed in the quantize layer
        - add powerful network before the MLP vector head?
    """

    def __init__(self,
                 vector_decoder_model: str = "mlp",
                 quantized_dim: int = 256,
                 codebook_size: int = 512,
                 patch_size:int=128,
                 image_loss: str = "pyramid",
                 num_codes_per_shape: int = 1,
                 vq_method:str = "fsq",
                 fsq_levels:list =[8,5,5,5],
                 num_segments:int = 1,
                 geometric_constraint: str = None,
                 geometric_constraint_weight: float = 0.1,
                 pred_color:bool=False,
                 **kwargs) -> None:
        super(Vector_VQVAE, self).__init__()

        assert vector_decoder_model in ["mlp", "raster_conv", "cnn"], "vector_decoder_model must be one of ['mlp', 'raster_conv', 'cnn']"
        assert geometric_constraint in ["inner_distance", None], f"geometric_constraint must be one of ['inner_distance'], but was {geometric_constraint}"

        self.vector_decoder_model = vector_decoder_model
        self.quantized_dim = quantized_dim
        self.image_loss = image_loss
        self.vq_method = vq_method.lower()
        assert self.vq_method == "fsq", "Please use FSQ."
        self.fsq_levels = fsq_levels
        self.num_segments = num_segments
        self.num_codes_per_shape = num_codes_per_shape
        self.pred_color = pred_color
        self.patch_size = patch_size

        if geometric_constraint is not None:
            self.geometric_constraint = geometric_constraint
            self.geometric_constraint_weight = geometric_constraint_weight
        else:
            self.geometric_constraint = "None"
            self.geometric_constraint_weight = 0.0

        if self.vq_method == "fsq":
            self.codebook_size = np.prod(fsq_levels)
        else:
            self.codebook_size = codebook_size

        self.encoder = resnet18(num_classes = self.quantized_dim * self.num_codes_per_shape)

        if self.vq_method == "vqvae":
            self.quantize_layer = VectorQuantizer(num_embeddings = self.codebook_size,
                                                embedding_dim = self.quantized_dim,
                                                beta = 0.25)
        elif self.vq_method == "fsq":
            self.quantize_layer = FSQ(levels=self.fsq_levels,
                                      dim=self.quantized_dim)
        elif self.vq_method == "vqtorch":
            raise NotImplementedError("VQVAE with vqtorch not implemented yet.")
        else:
            raise ValueError(f"vq_method must be one of ['vqvae', 'fsq', 'vqtorch'], but is {self.vq_method}")

        
        self.latent_dim = self.quantized_dim
        
        if self.vector_decoder_model == "mlp":
            self.decoder = MLPVectorHeadFixed(latent_dim = self.quantized_dim * self.num_codes_per_shape,
                                              segments = self.num_segments,
                                              imsize = self.patch_size,
                                              max_stroke_width=20.)
        elif self.vector_decoder_model == "cnn":
            self.decoder = CNNVectorHead(latent_dim = self.quantized_dim * self.num_codes_per_shape,
                                        segments = self.num_segments,
                                        imsize = self.patch_size,
                                        max_stroke_width=20.,
                                        pred_color=self.pred_color,)
        elif self.vector_decoder_model == "raster_conv":
            self.decoder = DeconvResNet()

    def encode(self, input: Tensor, quantize: bool = False):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) latent codes
        """
        result = self.encoder.forward(input)  # output from default resnet pytorch is (bs, self.quantized_dim * self.num_codes_per_shape)
        while result.dim() < 4:
            result = result.unsqueeze(-1)
        if self.num_codes_per_shape > 1:
            result = rearrange(result, 'b (c2 c) h w -> b c2 (c h) w', c2=self.quantized_dim)
        # result = self.mapping_layer(result.view(-1, 512 * 4 * 4))
        if quantize:
            result = self.quantize_layer.forward(result)  # this might change the result return type to list
        return result
    
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result, logging_dict = self.decoder.forward(z)
        # if self.vector_decoder_model == "mlp":
        #     result = result[0]  # extract only the raster image for now
        return result, logging_dict
    
    def decode_from_indices(self, idxs: Tensor) -> Union[Tensor, dict]:
        """
        Maps the given idxs to [reconstructions, input, all_points, vq_loss], all_points are the points of the bezier curves
        :param z: (Tensor) [B x 1]

        """
        if self.vq_method == "fsq":
            codes = self.quantize_layer.indices_to_codes(idxs)
        else:
            raise NotImplementedError("Only FSQ implemented for now.")
        result, logging_dict = self.decoder.forward(codes)
        # if self.vector_decoder_model == "mlp":
        #     result = result[0]  # extract only the raster image for now
        return result, logging_dict
    
    
    def forward(self, input: Tensor, logging = False, return_widths = False, only_return_recons = False, **kwargs):
        logging_dict = {}
        encoding = self.encode(input, quantize=False)
        bs = encoding.shape[0]
        vq_logging_dict={}
        if self.vector_decoder_model in ["mlp", "cnn"]:
            # quantize the encoding
            if self.vq_method == "vqvae":
                quantized_inputs, vq_loss, vq_logging_dict = self.quantize_layer.forward(encoding, logging=logging)
            elif self.vq_method == "fsq":
                quantized_inputs, indices = self.quantize_layer.forward(encoding)
                vq_loss = torch.tensor(0.)
                if logging:
                    vq_logging_dict = {"codebook_histogram":wandb.Image(tensor_to_histogram_image(indices.detach().flatten().cpu()), caption="histogram of codebook indices")}
                    
            # flatten it for MLP digestion
            # quantized_inputs = quantized_inputs.permute(0,2,1,3)
            quantized_inputs = rearrange(quantized_inputs, 'b d (c h) w -> b (d c) h w', c=self.num_codes_per_shape)
            quantized_inputs = quantized_inputs.view(bs, self.quantized_dim * self.num_codes_per_shape)
            # print("quantized_inputs: ", quantized_inputs.shape)
        elif self.vector_decoder_model == "raster_conv":
            quantized_inputs, vq_loss = self.quantize_layer(encoding)
        
        # re-merge the quantized codes
        # quantized_inputs = rearrange(quantized_inputs, 'b d (c h) w -> b (d c) h w', c=self.num_codes_per_shape)
        out, decode_logging_dict = self.decode(quantized_inputs)  # for mlp out is [output, scenes, all_points, all_widths]
        reconstructions = out[0]
        all_points = out[2]
        all_widths = out[3]
        logging_dict = {**logging_dict, **decode_logging_dict, **vq_logging_dict}
        if only_return_recons:
            return reconstructions
        if return_widths:
            return [reconstructions, input, all_points, vq_loss, all_widths], logging_dict
        else:
            return [reconstructions, input, all_points, vq_loss], logging_dict
    
    def gaussian_pyramid_loss(self, recons_images: Tensor, gt_images: Tensor, down_sample_steps: int = 3, log_loss: bool = False):
        """
        Calculates the gaussian pyramid loss between reconstructed images and ground truth images.

        Args:
            - recons_images (Tensor): Reconstructed images in format (-1, c, w, h)
            - gt_images (Tensor): Ground truth images in format (-1, c, w, h)
            - down_sample_steps (int): Number of downsample steps to calculate the loss for. Default: 3

        Returns:
            - recon_loss (Tensor): The gaussian pyramid loss between reconstructed images and ground truth images.
        """
        dsample = kornia.geometry.transform.pyramid.PyrDown()
        timesteps_to_log = 4
        recon_loss = F.mse_loss(recons_images, gt_images, reduction='none')
        recons_loss_contributions = {}
        if log_loss:
            all_loss_images = []
            all_loss_images.append(self.transform_loss_tensor_to_image(recon_loss[:timesteps_to_log]))
        recon_loss = recon_loss.mean()
        for j in range(2, 2 + down_sample_steps):
            weight = 1 / j
            recons_images = dsample(recons_images)
            gt_images = dsample(gt_images)
            loss_images = F.mse_loss(recons_images, gt_images, reduction='none')
            if log_loss:
                all_loss_images.append(self.transform_loss_tensor_to_image(loss_images[:timesteps_to_log]))

            curr_pyramid_loss = loss_images.mean() / weight
            recons_loss_contributions[f"pyramid_loss_step_{j-1}"] = curr_pyramid_loss
            recon_loss = recon_loss + curr_pyramid_loss

        if log_loss:
            log_all_images(all_loss_images, log_key="pyramid loss", caption=f"Gaussian Pyramid Loss, {down_sample_steps+1} steps")
            wandb.log(recons_loss_contributions)
        return recon_loss

    def _get_mean_inner_distance(self,
                        points: Tensor) -> Tensor:
        """
        mean inner distance is defined as the distance between start and end point of each segment of the path
        """
        inner_dists = []
        # TODO experiment with quadratic distance here
        for i in range(self.num_segments):
            inner_dist = torch.cdist(points[:,:,i*3,:], points[:,:,(i+1)*3,:])
            inner_dists.append(inner_dist.mean())
        return torch.mean(torch.tensor(inner_dists))

    def loss_function(self,
                      reconstructions: Tensor,
                      gt_images: Tensor,
                      vq_loss: Tensor,
                      points: Tensor,
                      log_loss: bool = False,
                      **kwargs) -> dict:
        if self.image_loss == "mse":
            recons_loss = F.mse_loss(reconstructions, gt_images)
        elif self.image_loss == "pyramid":
            recons_loss = self.gaussian_pyramid_loss(reconstructions, gt_images, down_sample_steps=3, log_loss=log_loss)
        else:
            raise NotImplementedError("Only mse and pyramid loss implemented for now.")
        if self.geometric_constraint == "inner_distance":
            max_dist = torch.cdist(torch.tensor([[0.0,0.0]]), torch.tensor([[1.0,1.0]])).item()
            mean_inner_distance = self._get_mean_inner_distance(points)
            # loss is weighted by the mean of black pixels, so that short strokes are not penalized as much
            # TODO adapt to new loss formulation from thesis
            geometric_loss = (max_dist - mean_inner_distance) * (1- gt_images).mean()
        else:
            geometric_loss = 0.0
        
        loss = recons_loss + vq_loss + self.geometric_constraint_weight * geometric_loss

        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss,
                self.geometric_constraint+"_loss" : self.geometric_constraint_weight * geometric_loss}
    

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    
    @torch.no_grad()
    def reconstruct(self, patches: Tensor, gt_center_positions: Tensor, padded_individual_max_length: float, stroke_width: float, rendered_w = 128., return_shapes:bool=False) -> Union[Drawing, Tensor]:
        """
        Reconstructs the input patches and uses gt positions to assemble them into a full SVG. Can be used to observe quality degradation of the quantization process.
        TODO currently does not use the predicted stroke width but the GT one.
        
        Args:
            - patches (Tensor): Input patches to be reconstructed
            - gt_center_positions (Tensor): Ground truth center positions of the patches
            - padded_individual_max_length (float): Padded individual max length of the patches, usually is individual_max_length+2
            - stroke_width (float): Stroke width of the patches

        Returns:
            - reconstructed_drawing (Drawing): Reconstructed drawing (use to save svg)
            - rasterized_reconstructions (Tensor): Rasterized reconstructions
        """
        [reconstructions, input, all_points, vq_loss, all_widths], logging_dict = self.forward(patches, logging = False, return_widths=True)
        global_shapes = calculate_global_positions(all_points, padded_individual_max_length, gt_center_positions)[:,0]
        reconstructed_drawing = shapes_to_drawing(global_shapes, stroke_width=stroke_width, w=rendered_w)
        rasterized_reconstructions = svg_string_to_tensor(reconstructed_drawing.tostring())
        if return_shapes:
            return reconstructed_drawing, rasterized_reconstructions, global_shapes, all_widths
        else:
            return reconstructed_drawing, rasterized_reconstructions