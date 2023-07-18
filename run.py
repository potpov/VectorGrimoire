import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder, EarlyStopping
from thesis.dataset import MNISTDataset, MNISTppDataset, NounProjectDataset, EmojiDataset
import wandb

DATASETMAP = {
    "mnist" : MNISTDataset,
    "mnistpp" : MNISTppDataset,
    "nounproject" : NounProjectDataset,
    "emoji" : EmojiDataset
}


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
parser.add_argument(
    "--wandb",
    "-w",
    dest="wandb",
    help="want to log the run with wandb?",
    action=argparse.BooleanOptionalAction
)

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

if(args.wandb):
    wandb_logger = WandbLogger(name=config['logging_params']['name'], 
                               save_dir=config['logging_params']['save_dir'],
                               project = config["logging_params"]["project"],
                               log_model=True)
    wandb_logger.experiment.config.update(config)
else:
    wandb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['logging_params']['name'],)



# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

if(args.wandb):
    model = vae_models[config['model_params']['name']](**config['model_params'], wandb_logging=True)
    wandb.watch(model, log='all')
else:
    model = vae_models[config['model_params']['name']](**config['model_params'])

experiment = VAEXperiment(model,
                          config['exp_params'])

data = DATASETMAP[config["data_params"]["dataset"]](**config["data_params"], pin_memory=config['trainer_params']['devices'] > 0)

data.setup()
runner = Trainer(logger=wandb_logger,
                 callbacks=[
                     LearningRateMonitor(logging_interval="epoch", log_momentum=True),
                     LearningRateFinder(early_stop_threshold=None),
                     EarlyStopping("val_loss", 0.002, 4),
                     ModelCheckpoint(save_top_k=1, 
                                     dirpath =os.path.join(config['logging_params']['save_dir'] , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 **config['trainer_params'])


Path(f"{wandb_logger.save_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{wandb_logger.save_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)