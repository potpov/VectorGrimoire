from torchvision import transforms
from PIL import Image
import wandb
import numpy as np
import torch
import os
from torchvision.utils import make_grid
from torchvision.transforms import Resize


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer.buffer_rgba())
    return X[:,:,:3]


def make_tensor(x, grad=False):
    x = torch.tensor(x, dtype=torch.float32)
    x.requires_grad = grad
    return x


def log_images(recons, real_imgs, log_key="validation", captions="Captions not set"):

    if get_rank() != 0:
        return
    
    common_size = recons.shape[-2:]
    resizer = Resize(common_size)
    real_imgs_resized = resizer(real_imgs)

    image_result = torch.concat((
        make_grid(real_imgs_resized, nrow=4),
        make_grid(recons, nrow=4)
        ),
        dim=-1
    )

    wandb.log({log_key: wandb.Image(image_result, caption=captions)})


def get_rank() -> int:
    if not torch.distributed.is_available():
        return 0  # Training on CPU
    if not torch.distributed.is_initialized():
        rank = os.environ.get("LOCAL_RANK")  # from pytorch-lightning
        if rank is not None:
            return int(rank)
        else:
            return 0
    else:
        return torch.distributed.get_rank()
