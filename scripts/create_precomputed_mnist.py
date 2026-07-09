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


def exec(patch_size=128, tiles_per_row=6, total_padding=4, th_filter=0.2,
         source_dir=None, output_dir=None):
    # Paths are configurable (dead /raid defaults removed). Pass them explicitly,
    # or set MNIST_PNG_DIR / MNIST_PRETILED_DIR env vars. `output_dir` is the ROOT;
    # the P{...}_T{...}_P{...}_TH{...} subdir is appended automatically.
    source_dir = source_dir or os.environ.get("MNIST_PNG_DIR", "mnist_png")
    out_root = output_dir or os.environ.get("MNIST_PRETILED_DIR", "mnist_pretiled")
    new_dimension = (patch_size - total_padding) * tiles_per_row
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
        "patch_size": patch_size,
        "transform": my_transforms,
        "num_tiles_per_row": tiles_per_row,
        "total_padding": total_padding,
        "filter_th": th_filter,
        "source_dir": source_dir,
        "output_dir": os.path.join(out_root, f"P{patch_size}_T{tiles_per_row}_P{total_padding}_TH{th_filter}"),
    }
    Mnister(**config)


def create_yaml():
    """ make sure you are in the config file because this spanws the yaml files in the current directory"""
    import yaml
    import os
    import copy

    # Load the base YAML file
    with open("base.yaml", "r") as file:
        base_config = yaml.safe_load(file)

    # Define parameter ranges
    patch_sizes = [32, 64, 128, 256]
    n_tiles_list = [3, 5, 8]

    data_root = os.environ.get("MNIST_PRETILED_DIR", "mnist_pretiled")

    iteration = 0
    for patch_size in patch_sizes:
        for n_tiles in n_tiles_list:
            iteration += 1
            
            # Modify the configuration
            config = copy.deepcopy(base_config)
            config["data_params"]["patch_size"] = patch_size
            config["data_params"]["num_tiles_per_row"] = n_tiles
            config["data_params"]["data_path"] = os.path.join(data_root, f"P{patch_size}_T{n_tiles}_P4_TH0.2")
            
            config["logging_params"]["name"] = f"{iteration}_P{patch_size}_T{n_tiles}_P4_TH0.2"
            
            # Save the modified YAML
            output_filename = f"{iteration}.yaml"
            with open(output_filename, "w") as output_file:
                yaml.safe_dump(config, output_file, default_flow_style=False)
            
            print(f"Saved: {output_filename}")


if __name__ == '__main__':
    
    for patch_size in [32, 64, 128, 256]:
        for n_tiles in [3, 5, 8]:
            print(f"Executing with patch_size={patch_size} and tiles_per_row={n_tiles}")
            exec(patch_size=patch_size, tiles_per_row=n_tiles, total_padding=4, th_filter=0.2)




