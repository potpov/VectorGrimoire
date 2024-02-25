import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder, EarlyStopping
from thesis.dataset import MNISTDataset, MNISTppDataset, NounProjectDataset, EmojiDataset, MNISTDatasetCSVG, CausalSVGDataModule, NewCausalSVGDataModule, CenterShapeLayersFromSVGDataModule, VQDataModule, GlyphazznStage1Datamodule
from thesis.models import VAEctorGen, VectorGPT, VanillaVAE, VectorVAEnLayers, VectorGPTv2, Vector_VQVAE
from thesis.experiment import VAEXperiment, VectorGPTExperiment, VectorGPTExperimentv2, VectorVQVAE_Experiment_Stage1, SVG_VQVAE_Stage2_Experiment
import wandb
from thesis.utils import get_rank
import torch
from pytorch_lightning.profilers import SimpleProfiler

torch.set_float32_matmul_precision('high')

DATASETMAP = {
    "causalSVG": CausalSVGDataModule,
    "causalSVGv2": NewCausalSVGDataModule,
    "emoji": EmojiDataset,
    "nounproject": NounProjectDataset,
    "mnistpp": MNISTppDataset,
    "mnist": MNISTDataset,
    "mnistCSVG": MNISTDatasetCSVG,
    "centeredShapeLayers" : CenterShapeLayersFromSVGDataModule,
    "tokens" : VQDataModule,
    "stage1" : GlyphazznStage1Datamodule,
}

MODELS = {
    "VanillaVAE": VanillaVAE,
    "VAEctorGen": VAEctorGen,
    "VectorVAEnLayers": VectorVAEnLayers,
    "VectorGPT": VectorGPT,
    "VectorGPTv2": VectorGPTv2,
    "SVG_VAQVAE": Vector_VQVAE,
  }


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c', dest="filename", metavar='FILE', help='path to the config file', default='configs/vae.yaml')
parser.add_argument("--wandb", "-w", dest="wandb", action='store_true', help="want to log the run with wandb? (default false)")
parser.add_argument('--debug', action='store_true', help='disable wandb logs, set workers to 0. (default false)')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# assertions for the config file
if "context_length" in config["model_params"]:
    assert config["model_params"]["context_length"] == config["data_params"]["context_length"], f"context length in model and data params must be the same"
assert config["data_params"]["dataset"] in DATASETMAP.keys(), f"dataset {config['data_params']['dataset']} not supported, try one of {list(DATASETMAP.keys())}"
assert config["model_params"]["name"] in MODELS.keys(), f"model {config['model_params']['name']} not supported, try one of {list(MODELS.keys())}"

# disabling multi-threading when debugging
if args.debug:
    config["data_params"]["num_workers"] = 0

current_process_rank = get_rank()

if args.wandb:
    if "entity" not in config['logging_params']:
        entity = "mfeuer"
    else:
        entity = config['logging_params']['entity']
    wandb_logger = WandbLogger(
        name=config['logging_params']['name'],
        save_dir=config['logging_params']['save_dir'],
        tags=[config['logging_params']['author']],
        project=config["logging_params"]["project"],
        log_model=True,
        entity=entity,
        mode="disabled" if args.debug else "online",
    )
    if current_process_rank == 0:
        wandb_logger.experiment.config.update(config)
else:
    wandb_logger = TensorBoardLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name']
    )

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)
print("Loading model...")
if args.wandb:
    model = MODELS[config['model_params']['name']](**config['model_params'], wandb_logging=True)
    # wandb_logger.watch(model, log="gradients", log_freq=500, log_graph=False)
    # wandb.watch(model, log='all', log_freq=100)  # can be "all"
else:
    model = MODELS[config['model_params']['name']](**config['model_params'])
print("Loading dataset...")
if config['model_params']['name'] == "VectorGPT":
    data = DATASETMAP[config["data_params"]["dataset"]](**config["data_params"])
    experiment = VectorGPTExperiment(model, **config['exp_params'], wandb = args.wandb)
elif config['model_params']['name'] == "VectorGPTv2":
    data = DATASETMAP[config["data_params"]["dataset"]](**config["data_params"])
    experiment = VectorGPTExperimentv2(model, **config['exp_params'], wandb = args.wandb)
elif config['model_params']['name'] == "SVG_VAQVAE":
    data = DATASETMAP[config["data_params"]["dataset"]](**config["data_params"])
    experiment = VectorVQVAE_Experiment_Stage1(model, **config['exp_params'], wandb = args.wandb, datamodule = data)
elif config['model_params']['name'] == "VQ_Transformer":
    data = DATASETMAP[config["data_params"]["dataset"]](**config["data_params"], context_length = config["model_params"]["max_seq_len"])
    experiment = SVG_VQVAE_Stage2_Experiment(model, **config['exp_params'], wandb = args.wandb)
else:
    data = DATASETMAP[config["data_params"]["dataset"]](**config["data_params"])
    experiment = VAEXperiment(model, config['exp_params'])

print("Setting up data...")
data.setup()
print("Setting up trainer...")
profiler = SimpleProfiler(dirpath=os.path.join(config['logging_params']['save_dir']))
runner = Trainer(
    logger=wandb_logger,
    # strategy='ddp_find_unused_parameters_true',
    callbacks=[
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        #  LearningRateFinder(early_stop_threshold=None, num_training_steps=200),
        EarlyStopping("val_loss", 0.005, 5, verbose=True),
        ModelCheckpoint(save_top_k=2,
                        dirpath=os.path.join(config['logging_params']['save_dir'], "checkpoints"),
                        monitor="val_loss",
                        save_last=True,
                        every_n_train_steps=2000),
    ],
    #  overfit_batches=20,
    log_every_n_steps=max(int(config['exp_params']["train_log_interval"] / 10), 5),
    profiler=profiler,
    **config['trainer_params']
)


Path(f"{wandb_logger.save_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{wandb_logger.save_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
try:
    # Start training
    if os.path.exists(config["exp_params"]["continue_checkpoint"]):
        runner.fit(experiment, datamodule=data, ckpt_path=config["exp_params"]["continue_checkpoint"])
    else:
        runner.fit(experiment, datamodule=data)
    profiler.describe()
    print(profiler.summary())
    with open("profiler_results.txt", "w") as f:
        f.write(profiler.summary())
except KeyboardInterrupt:
    # Handle the interrupt and save the profiling results
    print("Training interrupted by user.")
    profiler.describe()
    print(profiler.summary())
    with open("profiler_results.txt", "w") as f:
        f.write(profiler.summary())

