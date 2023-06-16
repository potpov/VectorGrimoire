import yaml
import torch
from torch import Tensor
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import os
from models import VanillaVAE

def _load_vanilla_vae(config_path: str, weights_path:str, device:str = "cuda:0"):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            

    model = VanillaVAE(**config['model_params'])
    state_dict = torch.load(weights_path)["state_dict"]

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model.to(device)
    return model

def _load_dataset_into_tensor(images_path:str, img_size:int=64)->Tensor:
    """
    returns dataset as resized and grayscaled Tensor in (B x C x W x H)-format
    """
    tensors=[]
    transforms_ = transforms.Compose(
        transforms=[
            transforms.Resize(img_size),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ]
    )
    for path in os.listdir(images_path):
        absolute_path = os.path.join(images_path, path)
        image = Image.open(absolute_path)
        if(image.mode == "RGBA"):
                bg = Image.new("RGB", image.size, (255,255,255))
                bg.paste(image, mask=image.split()[3])
                image = bg
        tensors.append(transforms_(image))

    full_dataset = torch.stack(tensors, 0)
    return full_dataset

def _display_images(images:Tensor, nrow:int=5, title=None):
    """
    images need to be in (B x C x W x H) format
    """
    grid_image = vutils.make_grid(images, normalize=True, nrow=nrow)
    pil_image = transforms.ToPILImage()(grid_image)
    plt.imshow(pil_image)
    if(title):
        plt.title(title)
    plt.axis('off')
    plt.show()