from tqdm import tqdm
import yaml
from thesis.dataset import GlyphazznStage1Datamodule



with open('/home/mfeuerpfeil/master/thesis/configs/SVG_VQVAE_Stage1_figr8.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config["data_params"]["num_workers"] = 32
config["data_params"]["train_batch_size"] = 32
config["data_params"]["val_batch_size"] = 32


train_dm = GlyphazznStage1Datamodule(**config["data_params"])
train_dm.setup()
train_dl = train_dm.train_dataloader()
for batch in tqdm(train_dl, total = len(train_dl)):
    imgs, lables, centers, descr = batch

print("DONE")
