from .base import *
from .vanilla_vae import *
from .vq_vae import *
from .vector_vae_nlayers import *
from thesis.models.vaectorgen import VAEctorGen

vae_models = {"VanillaVAE": VanillaVAE, 
              "VQVAE": VQVAE,
              "Im2Vec" : VectorVAEnLayers,
              "VAEctorGen" : VAEctorGen}
