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
from dataset import MNISTDataset, MNISTppDataset, NounProjectDataset, EmojiDataset, MNISTDatasetCSVG, CausalSVGDataModule
from models import VAEctorGen, VectorGPT, VanillaVAE, VectorVAEnLayers
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

MODELS = {
    "VanillaVAE": VanillaVAE,
    "VAEctorGen": VAEctorGen,
    "VectorVAEnLayers": VectorVAEnLayers,
    "VectorGPT": VectorGPT,
  }


# parser = argparse.ArgumentParser(description='Generic runner for VAE models')
# parser.add_argument('--config',  '-c', dest="filename", metavar='FILE', help='path to the config file', default='configs/vae.yaml')


# args = parser.parse_args()
# with open(args.filename, 'r') as file:
#     try:
#         config = yaml.safe_load(file)
#     except yaml.YAMLError as exc:
#         print(exc)

# # assertions for the config file
# if "context_length" in config["model_params"]:
#     assert config["model_params"]["context_length"] == config["data_params"]["context_length"], f"context length in model and data params must be the same"
# assert config["data_params"]["dataset"] in DATASETMAP.keys(), f"dataset {config['data_params']['dataset']} not supported, try one of {list(DATASETMAP.keys())}"
# assert config["model_params"]["name"] in MODELS.keys(), f"model {config['model_params']['name']} not supported, try one of {list(MODELS.keys())}"

def load_default_config():
    with open('configs/VectorGPT_sweep_basis.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def get_sweep_config():
    sweep_config = {
        'method': 'random',
        "metric": {
            'name': 'loss',
            'goal': 'minimize'   
        },
        'parameters': {
            # model_params
            'learnable_positional_encoding': {
                'values': [True, False]
            },
            'skip_transformer': {
                'values': [True, False]
            },
            'latent_transformer_dim': {
                'values': [32, 64, 128, 256]
            },
            'latent_transformer_heads': {
                'values': [4, 8, 16]
            },
            'latent_transformer_depth': {
                'values': [2, 4, 6, 8]
            },
            'vector_decoder_model': {
                'values': ['mlp', 'cnn']
            },
            'vector_decoder_paths': {
                'values': [1, 2, 4]
            },
            'vector_decoder_filled': {
                'values': [True, False]
            },
            'vector_decoder_max_stroke_width': {
                'values': [10.0, 15.0]
            },
            "loss_mode" : {
                "values": ["pyramid", "merged", "pyramid+merged"]
            },
            "down_sample_steps": {
                "values": [3, 4, 5]
            },
            # exp_params
            'input_mode': {
                'values': ['layer', 'merged']
            },
            'lr': {
                'distribution': 'uniform',
                'min': 1.0e-5,
                'max': 5.0e-3
            },
            'scheduler_gamma': {
                'values': [None, 0.95, 0.99]
            }
        }
    }
    return sweep_config


default_config = load_default_config()
sweep_config = get_sweep_config()
sweep_id = wandb.sweep(sweep_config, project="test")

# train wrapper for sweep agents
def train(config = None):
    with wandb.init(config=sweep_config):
        config = wandb.config
        # update config with selected sweep values
        name_of_run = "VectorGPT_OpenMoji_"
        for key in config.keys():
            if key in ["method", "metric", "parameters"]:
                continue
            if key not in ["input_mode", "lr", "scheduler_gamma"]:
                default_config["model_params"][key] = config[key]
            else:
                default_config["exp_params"][key] = config[key]
            name_of_run += f"{key}={config[key]}_"

        default_config["logging_params"]["name"] = name_of_run

    config = default_config
    if config["exp_params"]["scheduler_gamma"] is None:
        config["exp_params"]["weight_decay"] = 0.0
    config["model_params"]["vector_decoder_latent_dim"] =  config["model_params"]["latent_transformer_dim"]
    # print(config)

    current_process_rank = get_rank()

    wandb_logger = WandbLogger(
        name=config['logging_params']['name'],
        save_dir=config['logging_params']['save_dir'],
        tags=[config['logging_params']['author']],
        project=config["logging_params"]["project"],
        log_model=False,
        entity="aiis-chair",
        mode="online",
    )
    if current_process_rank == 0:
        wandb_logger.experiment.config.update(config)


    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)


    model = MODELS[config['model_params']['name']](**config['model_params'], wandb_logging=True)
    wandb_logger.watch(model, log="gradients", log_freq=50, log_graph=False)
    # wandb.watch(model, log='all', log_freq=100)  # can be "all"

    if config['model_params']['name'] == "VectorGPT":
        experiment = VectorGPTExperiment(model, **config['exp_params'], wandb = True)
    else:    
        experiment = VAEXperiment(model, config['exp_params'])

    data = DATASETMAP[config["data_params"]["dataset"]](**config["data_params"], pin_memory=True)

    data.setup()
    runner = Trainer(
        logger=wandb_logger,
        # strategy='ddp_find_unused_parameters_true',
        callbacks=[
            LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        ],
        #  overfit_batches=20,
        log_every_n_steps=max(int(config['exp_params']["train_log_interval"] / 10), 5),
        **config['trainer_params']
    )


    Path(f"{wandb_logger.save_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{wandb_logger.save_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)

# ----------------------
# ------ ACTION --------
# ----------------------

if __name__ == "__main__":
    wandb.agent(sweep_id, train, count=25)