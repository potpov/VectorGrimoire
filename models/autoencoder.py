import torch
from torch import nn
import torch.nn.functional as F

from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

class AutoEncoder(nn.Module):
    """
    Autoencoder to pre-train the image encoder on the domain of the dataset.

    Args:
        - image_encoder_model (str): Name of the image encoder model to use. Available options are: "resnet18", "resnet34", "resnet50", "resnet101", "resnet152".
        - latent_dim (int): Dimension of the latent representation. default: 128
    """
    def __init__(self, 
                 image_encoder_model: str = "resnet18",
                 latent_dim = 128,
                 use_fc = True):
        
        super(AutoEncoder, self).__init__()

        self.image_encoder_model = image_encoder_model
        self.latent_dim = latent_dim
        self.use_fc = use_fc

        if self.image_encoder_model == "resnet18":
            self.resnet = ResNet18(self.latent_dim)
        elif self.image_encoder_model == "resnet34":
            self.resnet = ResNet34(self.latent_dim)
        elif self.image_encoder_model == "resnet50":
            self.resnet = ResNet50(self.latent_dim)
        elif self.image_encoder_model == "resnet101":
            self.resnet = ResNet101(self.latent_dim)
        elif self.image_encoder_model == "resnet152":
            self.resnet = ResNet152(self.latent_dim)
        else:
            raise ValueError(f"[ERROR] You did not specify a correct Image Encoder. Expected something like 'resnet18', got {self.image_encoder_model}.")

        self.upscale_latent_dim = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # this configuration outputs twice the input width
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2),
            nn.ConvTranspose2d(2, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048+2048),
            nn.ReLU(),
            nn.Linear(2048+2048, 4096*2),
            nn.ReLU(),
            nn.Linear(4096*2, 1 * 128 * 128),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.resnet(x)
        return x

    def decode(self, x):
        if self.use_fc:
            x = self.fc_decoder(x)
            x = x.view(-1, 1, 128, 128)
        else:
            x = self.upscale_latent_dim(x)
            x = x.view(-1, 64, 2, 2)
            x = self.decoder(x)

        x = x.repeat(1, 3, 1, 1)  # repeat grayscale image to have 3 channels
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def loss_function(self, x, x_hat):
        """
        Loss function for the autoencoder. Returns the MSE loss.
        """
        return F.mse_loss(x_hat, x)