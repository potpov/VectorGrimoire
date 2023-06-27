from torchvision import transforms
from PIL import Image
import wandb
import numpy as np
import torch

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

def log_images(recons, real_imgs, log_key="validation", captions=None):
    if captions is not None:
        assert len(captions) == len(
            recons
        ), "shapes of captions and reconstructions do not match"

    def get_concat_h(im1, im2):
        max_heigth = np.max([im1.height, im2.height])
        dst = Image.new("RGB", (im1.width + im2.width, max_heigth), color="white")
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    try:
        # try to log all validation images
        input_imgs = [
            transforms.ToPILImage()(real_imgs[i]).convert("RGB")
            for i in range(len(real_imgs))
        ]
        recons_imgs = [
            transforms.ToPILImage()(recons[i]).convert("RGB")
            for i in range(len(recons))
        ]

        combined_imgs = [
            get_concat_h(input_imgs[i], recons_imgs[i]) for i in range(len(input_imgs))
        ]
        if captions:
            wandb.log(
                {
                    log_key: [
                        wandb.Image(combined_imgs[i], caption=captions[i])
                        for i in range(len(combined_imgs))
                    ]
                }
            )
        else:
            wandb.log(
                {
                    log_key: [
                        wandb.Image(combined_imgs[i]) for i in range(len(combined_imgs))
                    ]
                }
            )

    except:
        # when fails, try to log at least the first one
        try:
            input_img = transforms.ToPILImage()(real_imgs[0]).convert("RGB")
            recons_img = transforms.ToPILImage()(recons[0]).convert("RGB")

            combined_img = get_concat_h(input_img, recons_img)
            wandb.log({log_key: wandb.Image(combined_img)})
        except:
            pass
