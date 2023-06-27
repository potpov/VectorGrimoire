import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from thesis.models import BaseVAE


class VAEctorGen(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: list = None,
        **kwargs
    ) -> None:
        
        super(VAEctorGen, self).__init__()

        self.latent_dim = latent_dim

        # CNN encoder, calculate with 128 img size
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Mean and variance of distribution
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        self._init_embeddings()

        # Build Decoder
        
        pass

    def _init_embeddings(self):
        nn.init.normal_(self.fc_mu.weight, std=0.001)
        nn.init.constant_(self.fc_mu.bias, 0)
        nn.init.normal_(self.fc_var.weight, std=0.001)
        nn.init.constant_(self.fc_var.bias, 0)

    def forward(
        self,
    ):
        # encode image through encoder

        # get mean and var

        # reparameterization trick

        # feed noise vector to Transformer together with embeddings
        # 1. embedd noise vector with FF
        pass
