from typing import List
from torchvision import transforms
from PIL import Image
import wandb
import numpy as np
import torch
from torch import Tensor
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

def log_all_images(images: List[Tensor], log_key="validation", caption="Captions not set"):
    """
    Logs all images of a list as grids to wandb.

    Args:
        - images (List[Tensor]): List of images to log
        - log_key (str): key for wandb logging
        - captions (str): caption for the images
    """
    if get_rank() != 0:
        return

    assert len(images) > 0, "No images to log"

    common_size = images[0].shape[-2:]
    resizer = Resize(common_size)

    image_result = make_grid(images[0], nrow=4)
    for image in images[1:]:
        image_result = torch.concat((image_result, make_grid(resizer(image), nrow=4)), dim=-1)

    wandb.log({log_key: wandb.Image(image_result, caption=caption)})

def log_images(recons: Tensor, real_imgs: Tensor, log_key="validation", captions="Captions not set"):

    if get_rank() != 0:
        return

    if recons.shape[-2:] != real_imgs.shape[-2:]:
        common_size = recons.shape[-2:]
        resizer = Resize(common_size)
        real_imgs_resized = resizer(real_imgs)
    else:
        real_imgs_resized = real_imgs

    bs, c, w, h = real_imgs_resized.shape

    if recons.shape[1] > real_imgs_resized.shape[1]:
        real_imgs_resized = torch.cat((real_imgs_resized, torch.ones((bs, 1, w, h), device=real_imgs_resized.device)), dim=1)
    elif recons.shape[1] < real_imgs_resized.shape[1]:
        recons = torch.cat((recons, torch.ones((bs, 1, w, h), device=recons.device)), dim=1)

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
