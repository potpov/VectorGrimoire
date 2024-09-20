import os
import yaml
import argparse
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder, EarlyStopping
from dataset import VQDataModule
from models import VQ_SVG_Stage2, VSQ
from tokenizer import RasterVQTokenizer
from experiment import SVG_VQVAE_Stage2_Experiment
import json
from utils import get_rank
import torch
from pytorch_lightning.profilers import SimpleProfiler

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c', dest="filename", metavar='FILE', help='path to the config file', default='configs/vae.yaml')
parser.add_argument("--wandb", "-w", dest="wandb", action='store_true', help="want to log the run with wandb? (default false)")
parser.add_argument("--wandb_id", "-w_id", dest="wandb_id", type=int, default=None, help="id of wandb run to continue")
parser.add_argument('--debug', action='store_true', help='disable wandb logs, set workers to 0. (default false)')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

if "continue_checkpoint" in config["exp_params"] and config["exp_params"]["continue_checkpoint"] is not None:
    assert os.path.exists(config["exp_params"]["continue_checkpoint"]), f"checkpoint {config['exp_params']['continue_checkpoint']} does not exist"
    print(f"Found checkpoint to continue training from: {config['exp_params']['continue_checkpoint']}")
    if not args.wandb_id:
        print(f"wandb id must be set in logging_params to continue the logging in wandb")
        input("Press Enter to continue without continuing in wandb or CTRL+C to cancel")
else:
    assert not args.wandb_id, f"wandb id must not be set if not continuing from a checkpoint"

# disabling multi-threading when debugging
if args.debug:
    config["data_params"]["num_workers"] = 0

current_process_rank = get_rank()
config['logging_params']['save_dir'] = os.path.join(
    config['logging_params']['save_dir'],
    config['logging_params']['name']
)
print(f"Updated configuration to log in: {config['logging_params']['save_dir']}")
Path(config['logging_params']['save_dir']).mkdir(exist_ok=True, parents=True)

# dumping config file
with open(os.path.join(config['logging_params']['save_dir'], 'config.json'), 'w') as f:
    json.dump(config, f)

if args.wandb:
    if "entity" not in config['logging_params']:
        entity = "aiis-chair"
    else:
        entity = config['logging_params']['entity']
    wandb_logger = WandbLogger(
        name=config['logging_params']['name'],
        save_dir=config['logging_params']['save_dir'],
        tags=[config['logging_params']['author']],
        project=config["logging_params"]["project"],
        log_model=True,
        entity=entity,
        mode="offline" if args.debug else "online",
        resume="must" if "continue_checkpoint" in config["exp_params"] else "allow",
        id=args.wandb_id  # default None -> start a new run
    )
    if current_process_rank == 0:
        allow_val_change = True if config["logging_params"].get("allow_val_change") else False
        wandb_logger.experiment.config.update(config, allow_val_change=allow_val_change)
else:
    wandb_logger = TensorBoardLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name']
    )

# Load auxiliary models
vq_model = VSQ(patch_size=config['data_params']["patch_size"], **config['stage1_params'], device = device)
state_dict = torch.load(config['stage1_params']["checkpoint_path"], map_location=device)["state_dict"]
try:
    vq_model.load_state_dict(state_dict)
except:
    vq_model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
vq_model = vq_model.eval()
tokenizer = RasterVQTokenizer(vq_model, 
                                tokens_per_patch=1,
                                do_tokenize_positions=False,
                                patch_size=config['data_params']["patch_size"],
                                num_tiles_per_row=config['data_params']["num_tiles_per_row"],
                                device=device,
                                use_text_encoder_only=False
                              )



# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)
model_name = config['model_params'].pop('name')
print(f"Loading model {model_name}...")
model = VQ_SVG_Stage2(tokenizer, **config['model_params'], device=device)

print("Loading dataset...")
data = VQDataModule(tokenizer=tokenizer,
                    **config["data_params"], 
                    context_length=config['model_params']['max_seq_len'])

print("Setting up data...")
data.setup()
print("Setting up trainer...")

num_batches_train = len(data.train_dataloader())
num_batches_val = len(data.val_dataloader())

experiment = SVG_VQVAE_Stage2_Experiment(model, 
                                         tokenizer, 
                                         **config['exp_params'], 
                                         wandb = args.wandb, 
                                         num_batches_train=num_batches_train, 
                                         num_batches_val=num_batches_val)

profiler = SimpleProfiler(dirpath=os.path.join(config['logging_params']['save_dir']))
runner = Trainer(
    logger=wandb_logger,
    callbacks=[
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        EarlyStopping("val_loss", 0.005, 5, verbose=True),
        ModelCheckpoint(save_top_k=3,
                        dirpath=os.path.join(config['logging_params']['save_dir'], "checkpoints"),
                        monitor="val_loss",
                        save_last=True),
    ],
    profiler=profiler,
    strategy="ddp_find_unused_parameters_true",
    **config['trainer_params']
)


Path(f"{wandb_logger.save_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{wandb_logger.save_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {model_name} =======")
try:
    # Start training
    if "continue_checkpoint" in config["exp_params"] and os.path.exists(config["exp_params"]["continue_checkpoint"]):
        runner.fit(experiment, datamodule=data, ckpt_path=config["exp_params"]["continue_checkpoint"])
        print(f"[INFO] Successfully loaded checkpoint from {config['exp_params']['continue_checkpoint']}.")
    else:
        print("[INFO] Started training from scratch.")
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
