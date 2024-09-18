"""
This script is used to preprocess the MNIST PNG into patches of a fixed size
and store it tensor files for the PreprocessingDataModule to load.
"""
import os
import torch
import torch.nn.functional as F
from PIL import Image
from utils import get_filter_function
import torchvision.transforms as transforms
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt


class Mnister:

    def __init__(self,
                 source_dir: str,
                 output_dir: str,
                 patch_size: int = 128,
                 transform=None,
                 num_tiles_per_row: int = 5,
                 total_padding: int = 20,
                 filter_th: float = 0,
                 ):

        super(Mnister, self)
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.transform = transform
        self.patch_size = patch_size
        self.num_tiles_per_row = num_tiles_per_row
        self.total_padding = total_padding
        self.th = filter_th

        self.output_dir = output_dir
        if os.path.exists(os.path.join(self.output_dir, "train.pt")):
            print(f"train patch file exists in {self.output_dir}")
            exit(0)
        else:
            pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            print("Pre-computing all patches and saving...")
            self.compute(is_train=True)
            self.compute(is_train=False)

    def compute(self, is_train):
        # the > make it work for threshold 0 as well because if there are 0 black
        # pixels then the sum is equal to the count of total pixels in the patch, not greater
        # alternative for the 0-th is:
        # filter_fn = lambda patches: patches[torch.any(patches != 1., dim=(1, 2, 3))]
        num_digits = 10
        filter_fn = get_filter_function(self.th, parse_patches=True)
        image_folder = os.path.join(self.source_dir, "training" if is_train else "testing")
        output_dir = os.path.join(self.output_dir, "train.pt" if is_train else "test.pt")
        samples = []
        for label in tqdm(range(num_digits), total=num_digits):
            label_folder = os.path.join(image_folder, str(label))
            image_files = os.listdir(label_folder)
            for image_file in tqdm(image_files, total=len(image_files), desc=f"patching {str(label)}", leave=False):
                image_path = os.path.join(label_folder, image_file)
                image = Image.open(image_path)
                if self.transform is not None:
                    image = self.transform(image)

                image = torch.where(image > 0.6, 1., 0.)  # makes binary
                patches = self.make_patches(image)

                # DEBUG
                # debug = patches.clone()
                # filter_idx = get_filter_function(self.th, parse_patches=False)
                # self.check_patch(debug)
                # debug[~filter_idx(patches)] = torch.ones((3, 128, 128))
                # self.check_patch(debug)

                patches = filter_fn(patches)  # removing empty patches
                # check recon with blank
                samples += list(patches)

        print("Saving patches in ", output_dir)
        torch.save(torch.stack(samples), output_dir)

    def make_patches(self, image):
        patches = []
        single_side_padding = self.total_padding // 2
        for i in range(0, image.shape[1], self.patch_size - single_side_padding * 2):
            for j in range(0, image.shape[2], self.patch_size - single_side_padding * 2):
                patch = image[:, i: i + self.patch_size - self.total_padding, j: j + self.patch_size - self.total_padding]
                patch = F.pad(patch, (single_side_padding, single_side_padding, single_side_padding, single_side_padding),
                              value=1.)
                patches.append(patch)
        return torch.stack(patches)

    def check_patch(self, patches):
        fig, axs = plt.subplots(self.num_tiles_per_row, self.num_tiles_per_row, figsize=(15, 15))
        for i in range(0, self.num_tiles_per_row):
            for j in range(0, self.num_tiles_per_row):
                axs[i, j].imshow(patches[i * self.num_tiles_per_row + j].permute(1, 2, 0).numpy())
                axs[i, j].axis('off')
            fig.tight_layout()
        fig.show()


if __name__ == '__main__':

    PATCH_SIZE = 128
    TILES_PER_ROW = 14
    TOTAL_PADDING = 20
    TH_FILTER = 0.2
    new_dimension = (PATCH_SIZE - TOTAL_PADDING) * TILES_PER_ROW
    print(f"New dimension with this configuration is: {new_dimension}x{new_dimension}.")

    my_transforms = transforms.Compose(
        [
            transforms.Resize(new_dimension, antialias=True),
            transforms.RandomInvert(1.0),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )


    config = {
        "patch_size": PATCH_SIZE,
        "transform": my_transforms,
        "num_tiles_per_row": TILES_PER_ROW,
        "total_padding": TOTAL_PADDING,
        "filter_th": TH_FILTER,
        "source_dir": "/raid/marco.cipriano/data/SVG/Grimoire/MNIST/mnist_png",
        "output_dir": f"/raid/marco.cipriano/data/SVG/Grimoire/MNIST/mnist_pretiled/P{PATCH_SIZE}_T{TILES_PER_ROW}_P{TOTAL_PADDING}_TH{TH_FILTER}",
    }
    Mnister(**config)
