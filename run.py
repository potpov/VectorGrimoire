import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from experiment import VAEXperiment, VectorGPTExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder, EarlyStopping
from thesis.dataset import MNISTDataset, MNISTppDataset, NounProjectDataset, EmojiDataset, MNISTDatasetCSVG, CausalSVGDataModule
from thesis.models import VAEctorGen, VectorGPT, VanillaVAE, VectorVAEnLayers
import wandb
from utils import get_rank
import torch


torch.set_float32_matmul_precision('high')

DATASETMAP = {
    "causalSVG": CausalSVGDataModule,
    "emoji": EmojiDataset,
    "nounproject": NounProjectDataset,
    "mnistpp": MNISTppDataset,
    "mnist": MNISTDataset,
    "mnistCSVG": MNISTDatasetCSVG
}

MODELS = {'VanillaVAE':VanillaVAE,
              'VAEctorGen':VAEctorGen,
              'VectorVAEnLayers': VectorVAEnLayers,
              "VectorGPT" : VectorGPT,
              }


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c', dest="filename", metavar='FILE', help='path to the config file', default='configs/vae.yaml')
parser.add_argument("--wandb", "-w", dest="wandb", help="want to log the run with wandb?", action=argparse.BooleanOptionalAction)
parser.add_argument('--debug', action='store_true', help='disable wandb logs, set workers to 0. (default false)')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# disabling multi-threading when debugging
if args.debug:
    config["data_params"]["num_workers"] = 0

if args.wandb and get_rank() == 0:
    wandb_logger = WandbLogger(
        name=config['logging_params']['name'],
        save_dir=config['logging_params']['save_dir'],
        tags=config['logging_params']['author'],
        project=config["logging_params"]["project"],
        log_model=True,
        entity="aiis-chair",
        mode="disabled" if args.debug else "online",
    )
    wandb_logger.experiment.config.update(config)
else:
    wandb_logger = TensorBoardLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name']
    )

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

if args.wandb and get_rank() == 0:
    model = MODELS[config['model_params']['name']](**config['model_params'], wandb_logging=True)
    wandb.watch(model, log='all', log_freq = 100) # can be "all"
else:
    model = MODELS[config['model_params']['name']](**config['model_params'])

if(config['model_params']['name'] == "VectorGPT"):
    experiment = VectorGPTExperiment(model, **config['exp_params'])
else:    
    experiment = VAEXperiment(model, config['exp_params'])

data = DATASETMAP[config["data_params"]["dataset"]](**config["data_params"], pin_memory=True, context_length = config['model_params']["context_length"])

data.setup()
runner = Trainer(logger=wandb_logger,
                 callbacks=[
                     LearningRateMonitor(logging_interval="epoch", log_momentum=True),
                     #  LearningRateFinder(early_stop_threshold=None, num_training_steps=200),
                     #  EarlyStopping("val_loss", 0.002, 3),
                     ModelCheckpoint(save_top_k=1, 
                                     dirpath =os.path.join(config['logging_params']['save_dir'], "checkpoints"),
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                #  overfit_batches=1,
                 **config['trainer_params'])


Path(f"{wandb_logger.save_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{wandb_logger.save_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)