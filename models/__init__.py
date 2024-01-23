from .base import *
from .vanilla_vae import *
from .vq_vae import *
from .vector_vae_nlayers import *
from models.vaectorgen import VAEctorGen
from .vectorGPT import VectorGPT
from .vectorGPTv2 import VectorGPTv2
from .svg_vqvae import Vector_VQVAE, VQ_Transformer

vae_models = {"VanillaVAE": VanillaVAE, 
              "VQVAE": VQVAE,
              "Im2Vec" : VectorVAEnLayers,
              "VAEctorGen" : VAEctorGen}
