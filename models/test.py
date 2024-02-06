from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import wandb
from thesis.utils import log_all_images
from models.resnet import ResNet, BasicBlock
from models.vq_vae import VectorQuantizer
from models.mlp_vector_head import MLPVectorHeadFixed
from models.mlp import MultiLayerPerceptron
from models.svg_vqvae import Vector_VQVAE

model = Vector_VQVAE(vector_decoder_model="mlp")
data = torch.randn(2, 3, 128, 128)
out = model.forward(data)