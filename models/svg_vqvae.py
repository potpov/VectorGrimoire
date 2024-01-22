from typing import Union
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import wandb
from utils import log_all_images, tensor_to_histogram_image
from models.resnet import ResNet, BasicBlock
from models.vq_vae import VectorQuantizer
from models.mlp_vector_head import MLPVectorHeadFixed
from models.mlp import MultiLayerPerceptron
from vector_quantize_pytorch import FSQ

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
                 image_loss: str = "pyramid",
                 single_code_representation: bool = True,
                 vq_method:str = "vqvae",
                 fsq_levels:list =[8,5,5,5],
                 **kwargs) -> None:
        super(Vector_VQVAE, self).__init__()

        assert vector_decoder_model in ["mlp", "raster_conv"], "vector_decoder_model must be one of ['mlp', 'raster_conv']"

        self.vector_decoder_model = vector_decoder_model
        self.quantized_dim = quantized_dim
        self.image_loss = image_loss
        self.vq_method = vq_method.lower()
        self.fsq_levels = fsq_levels
        if self.vq_method == "fsq":
            self.codebook_size = np.prod(fsq_levels)
        else:
            self.codebook_size = codebook_size

        self.encoder = ResNet(BasicBlock,
                              [2, 2, 2, 2],
                              10,
                              skip_linear=True)  # outputs (b, 512, 4, 4) - final W x H essentially decides the number of quantized vectors that form a single image, here its 4*4=16
        if single_code_representation:
            self.encoder = nn.Sequential(self.encoder,
                                         nn.Conv2d(512, self.quantized_dim, kernel_size=4, stride=4, padding=0))  # no ReLU here, we want to keep the negative values for the quantization

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
            self.decoder = MLPVectorHeadFixed(latent_dim = self.latent_dim,
                                              segments = 1,
                                              imsize = 128,
                                              max_stroke_width=20.)
        elif self.vector_decoder_model == "raster_conv":
            self.decoder = DeconvResNet()

    def encode(self, input: Tensor, quantize: bool = False):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) latent codes
        """
        result = self.encoder.forward(input)
        # result = self.mapping_layer(result.view(-1, 512 * 4 * 4))
        if quantize:
            result = self.quantize_layer.forward(result)
        return result
    
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space and position prediction.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result, logging_dict = self.decoder.forward(z)
        # if self.vector_decoder_model == "mlp":
        #     result = result[0]  # extract only the raster image for now
        return result, logging_dict
    
    
    def forward(self, input: Tensor, logging = False, **kwargs):
        logging_dict = {}
        encoding = self.encode(input, quantize=False)
        bs = encoding.shape[0]
        vq_logging_dict={}
        if self.vector_decoder_model == "mlp":
            # quantize the encoding
            if self.vq_method == "vqvae":
                quantized_inputs, vq_loss, vq_logging_dict = self.quantize_layer.forward(encoding, logging=logging)
            elif self.vq_method == "fsq":
                quantized_inputs, indices = self.quantize_layer.forward(encoding)
                vq_loss = torch.tensor(0.)
                if logging:
                    vq_logging_dict = {"codebook_histogram":wandb.Image(tensor_to_histogram_image(indices.detach().flatten().cpu()))}
                    
            # flatten it for MLP digestion
            quantized_inputs = quantized_inputs.view(bs, self.latent_dim)
            # print("quantized_inputs: ", quantized_inputs.shape)
        elif self.vector_decoder_model == "raster_conv":
            quantized_inputs, vq_loss = self.quantize_layer(encoding)
        
        out, decode_logging_dict = self.decode(quantized_inputs)  # for mlp out is [output, scenes, all_points, all_widths]
        reconstructions = out[0]
        all_points = out[2]
        logging_dict = {**logging_dict, **decode_logging_dict, **vq_logging_dict}
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

    def loss_function(self,
                      reconstructions: Tensor,
                      gt_images: Tensor,
                      vq_loss: Tensor,
                      log_loss: bool = False,
                      **kwargs) -> dict:
        if self.image_loss == "mse":
            recons_loss = F.mse_loss(reconstructions, gt_images)
        elif self.image_loss == "pyramid":
            recons_loss = self.gaussian_pyramid_loss(reconstructions, gt_images, down_sample_steps=3, log_loss=log_loss)
        else:
            raise NotImplementedError("Only mse and pyramid loss implemented for now.")
        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}
    
    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    
    def _assemble_svg(self, svg_predictions: Tensor, center_positions: Tensor):
        """
        Assembles the SVG prediction from the transformer into a single SVG.
        """
        raise NotImplementedError("Not implemented yet.")