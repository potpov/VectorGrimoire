from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import wandb
from utils import log_all_images
from models.resnet import ResNet, BasicBlock
from models.vq_vae import VectorQuantizer
from models.mlp_vector_head import MLPVectorHeadFixed
from models.mlp import MultiLayerPerceptron

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

        return x
    
class Vector_VQVAE(nn.Module):
    """
    Vector quantized pre-training of an autoencoder for SVG primitives.
    
    Input/Output are shape layers, no positions. Positions are decoded using the transformer in Stage II.

    TODO:
        - reduce encoding dimension from (512, 4, 4) to something more reasonable like (512, 2, 2), this must also be changed in the quantize layer
        - add powerful network before the MLP vector head?
        - pyramid loss for vector mlp head
    """

    def __init__(self,
                 vector_decoder_model: str = "mlp",
                 quantized_dim: int = 512,
                 **kwargs) -> None:
        super(Vector_VQVAE, self).__init__()

        assert vector_decoder_model in ["mlp", "raster_conv"], "vector_decoder_model must be one of ['mlp', 'raster_conv']"

        self.vector_decoder_model = vector_decoder_model
        self.quantized_dim = quantized_dim

        self.encoder = ResNet(BasicBlock,
                              [2, 2, 2, 2],
                              10,
                              skip_linear=True)  # outputs (b, 512, 4, 4) - final W x H essentially decides the number of quantized vectors that form a single image, here its 4*4=16
        
        self.quantize_layer = VectorQuantizer(num_embeddings = 64,
                                              embedding_dim = self.quantized_dim,
                                              beta = 0.25)
        
        self.latent_dim = self.quantized_dim * 4 * 4  # 4*4 is the final W x H of the encoder
        
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
            result = self.quantize_layer(result)
        return result
    
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space and position prediction.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder.forward(z)
        if self.vector_decoder_model == "mlp":
            result = result[0]  # extract only the raster image for now
        return result
    
    
    def forward(self, input: Tensor, **kwargs):
        encoding = self.encode(input, quantize=False)
        bs = encoding.shape[0]
        if self.vector_decoder_model == "mlp":
            # quantize the encoding
            quantized_inputs, vq_loss = self.quantize_layer(encoding)
            # flatten it for MLP digestion
            quantized_inputs = quantized_inputs.view(bs, self.latent_dim)
            # print("quantized_inputs: ", quantized_inputs.shape)
        elif self.vector_decoder_model == "raster_conv":
            quantized_inputs, vq_loss = self.quantize_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]
    
    def loss_function(self,
                      reconstructions: Tensor,
                      gt_images: Tensor,
                      vq_loss: Tensor,
                      **kwargs) -> dict:

        recons_loss = F.mse_loss(reconstructions, gt_images)
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