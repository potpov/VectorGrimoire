import os
import yaml
import argparse
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder, EarlyStopping
from thesis.dataset import VQDataModule
from thesis.models import VQ_SVG_Stage2, Vector_VQVAE
from thesis.tokenizer import VQTokenizer
from thesis.experiment import SVG_VQVAE_Stage2_Experiment
import wandb
from thesis.utils import get_rank
import torch
from pytorch_lightning.profilers import SimpleProfiler

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c', dest="filename", metavar='FILE', help='path to the config file', default='configs/vae.yaml')
parser.add_argument("--wandb", "-w", dest="wandb", action='store_true', help="want to log the run with wandb? (default false)")
parser.add_argument('--debug', action='store_true', help='disable wandb logs, set workers to 0. (default false)')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

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

# Load auxiliary models
vq_model = Vector_VQVAE(**config['stage1_params'], device = device)
state_dict = torch.load(config['stage1_params']["checkpoint_path"])["state_dict"]
try:
    vq_model.load_state_dict(state_dict)
except:
    vq_model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
vq_model = vq_model.eval()
tokenizer = VQTokenizer(vq_model, config["data_params"]["width"], 1, "bert-base-uncased", device = device)



# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)
print("Loading model...")
if args.wandb:
    model = VQ_SVG_Stage2(tokenizer, **config['model_params'], wandb_logging=True, device = device)
    # wandb_logger.watch(model, log="gradients", log_freq=500, log_graph=False)
    # wandb.watch(model, log='all', log_freq=100)  # can be "all"
else:
    model = VQ_SVG_Stage2(tokenizer, **config['model_params'], device = device)

print("Loading dataset...")
experiment = SVG_VQVAE_Stage2_Experiment(model, tokenizer, **config['exp_params'], wandb = args.wandb)
data = VQDataModule(**config["data_params"], context_length=config['model_params']['max_seq_len'])

print("Setting up data...")
data.setup()
print("Setting up trainer...")
profiler = SimpleProfiler(dirpath=os.path.join(config['logging_params']['save_dir']))
runner = Trainer(
    logger=wandb_logger,
    callbacks=[
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        EarlyStopping("val_loss", 0.005, 5, verbose=True),
        ModelCheckpoint(save_top_k=3,
                        dirpath=os.path.join(config['logging_params']['save_dir'], "checkpoints"),
                        monitor="val_loss",
                        save_last=True,
                        every_n_train_steps=10000),
    ],
    log_every_n_steps=int(config['exp_params']["train_log_interval"]),
    profiler=profiler,
    **config['trainer_params']
)


Path(f"{wandb_logger.save_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{wandb_logger.save_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
try:
    # Start training
    runner.fit(experiment, datamodule=data)
    profiler.describe()
    print(profiler.summary())
    with open("profiler_results_stage2.txt", "w+") as f:
        f.write(profiler.summary())
except KeyboardInterrupt:
    # Handle the interrupt and save the profiling results
    print("Training interrupted by user.")
    profiler.describe()
    print(profiler.summary())
    with open("profiler_results_stage2.txt", "w+") as f:
        f.write(profiler.summary())

