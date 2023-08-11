import torch
import torch.nn as nn
from torch import Tensor
from x_transformers import Decoder
from thesis.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from thesis.models.simple_vector_decoder import SimpleVectorDecoder
from thesis.models.mlp import MultiLayerPerceptron
from dataclasses import dataclass, field

@dataclass
class ImageEncoderArgs:
    model: str = "resnet18"

@dataclass
class LatentTransformerArgs:
    dim: int = 512
    depth: int = 8
    heads: int = 8
    layer_dropout: float = 0.1

@dataclass
class SimpleVectorDecoderArgs:
    latent_dim: int = 512
    paths: int = 1
    radius: int = 3
    render_size: int = 128

@dataclass
class MultiLayerPerceptronArgs:
    dims: list = field(default_factory=lambda: [768, 512])
    activation: str = "relu"
    num_classes: int = 2

@dataclass
class VectorGPTArgs:
    image_encoder_args: ImageEncoderArgs = ImageEncoderArgs()
    latent_transformer_args: LatentTransformerArgs = LatentTransformerArgs()
    simple_vector_decoder_args: SimpleVectorDecoderArgs = SimpleVectorDecoderArgs()
    stop_predictor_args: MultiLayerPerceptronArgs = MultiLayerPerceptronArgs()

class VectorGPT(nn.Module):
    def __init__(self,
                 vector_gpt_config: VectorGPTArgs, 
                 ):
        super(VectorGPT, self).__init__()

        self.vector_gpt_config = vector_gpt_config
        self.image_encoder_config = self.vector_gpt_config.image_encoder_args
        self.latent_transformer_config = self.vector_gpt_config.latent_transformer_args
        self.simple_vector_decoder_config = self.vector_gpt_config.simple_vector_decoder_args
        self.stop_predictor_config = self.vector_gpt_config.stop_predictor_args

        if(self.image_encoder_config.model == "resnet18"):
            self.resnet = ResNet18(self.latent_transformer_config.dim)
        elif(self.image_encoder_config.model == "resnet34"):
            self.resnet = ResNet34(self.latent_transformer_config.dim)
        elif(self.image_encoder_config.model == "resnet50"):
            self.resnet = ResNet50(self.latent_transformer_config.dim)
        elif(self.image_encoder_config.model == "resnet101"):
            self.resnet = ResNet101(self.latent_transformer_config.dim)
        elif(self.image_encoder_config.model == "resnet152"):
            self.resnet = ResNet152(self.latent_transformer_config.dim)
        else:
            raise ValueError(f"[ERROR] You did not specify a correct Image Encoder. Expected something like 'resnet18', got {self.image_encoder_config.model}.")
        
        self.latent_transformer = nn.Sequential(Decoder(**self.latent_transformer_config.__dict__), 
                                                nn.LayerNorm(self.latent_transformer_config.dim),
                                                nn.Linear(self.latent_transformer_config.dim, self.latent_transformer_config.dim))
        self.vector_decoder = SimpleVectorDecoder(**self.simple_vector_decoder_config.__dict__)
        self.stop_predictor = MultiLayerPerceptron(input_dim=self.latent_transformer_config.dim, **self.stop_predictor_config.__dict__)

    def forward(self, images: Tensor, stop_signals: Tensor):
        """
        Expects images to be in (batch, timesteps, channel, width, height).

        Outputs rasterized images
        """
        bs = images.size(0)
        timesteps = images.size(1)

        # first we encode. (b, t, c, w, h) -> (b, t, z)
        intermediate = [self.resnet(images[:,t,:,:]) for t in range(timesteps)]
        encoded_images = torch.stack(intermediate, dim=1) # (b, t, z)

        # then we transform (b, t, z) -> (b, t, z')
        transformed_latents = self.latent_transformer(encoded_images)

        # then we decode each t iteratively
        for t in range(timesteps):
            pass
        # for each t (t') in (b, t, z')
            # if not stop 
                # predict image from (b, z')
                # add that to p
            # else
                # forward 



    def loss_function(self, images: Tensor, stop_signals:Tensor):
        pass