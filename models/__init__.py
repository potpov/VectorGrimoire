from .base import *
from .vanilla_vae import *
from .vq_vae import *
from .vector_vae_nlayers import *
from .vaectorgen import VAEctorGen
from .vectorGPT import VectorGPT
from .vectorGPTv2 import VectorGPTv2
from .svg_vqvae import Vector_VQVAE, VQ_Transformer_deprecated
from .stage2 import VQ_SVG_Stage2

vae_models = {"VanillaVAE": VanillaVAE,
              "VQVAE": VQVAE,
              "Im2Vec" : VectorVAEnLayers,
              "VAEctorGen" : VAEctorGen}
