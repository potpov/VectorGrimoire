import gc
import os
import math
import random
import torch
from torch import Tensor
from torch import optim
from models import BaseVAE, VectorVAEnLayers, VectorGPT, VectorGPTv2, Vector_VQVAE
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from utils import log_images
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore

class VectorVQVAE_Experiment_Stage1(pl.LightningModule):
    """
    Vector quantized pre-training of an autoencoder for SVG primitives.
    
    Input/Output are shape layers and positions.
    """

    def __init__(self,
                 model: Vector_VQVAE,
                 vector_decoder_model: str = "raster_conv",  # or mlp
                 lr: float = 0.0003,
                 weight_decay: float = 0.0,
                 scheduler_gamma: float = 0.99,
                 train_log_interval: int = 250,
                 manual_seed: int = 42,
                 wandb: bool = True,
                 **kwargs) -> None:
        super(VectorVQVAE_Experiment_Stage1, self).__init__()

        self.model = model
        self.vector_decoder_model = vector_decoder_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.train_log_interval = train_log_interval
        self.manual_seed = manual_seed
        self.curr_device = None
        self.wandb = wandb

    def forward(self, input_images: Tensor, **kwargs) -> list:
        return self.model.forward(input_images, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        all_center_shapes, label = batch  # TODO this has one dimension too much rn
        self.curr_device = all_center_shapes.device
        bs = all_center_shapes.shape[0]
        channels = all_center_shapes.shape[1]

        out = self.forward(all_center_shapes)
        reconstructions=out[0]
        inputs = all_center_shapes
        vq_loss=out[2]

        loss_dict = self.model.loss_function(
            reconstructions=reconstructions[:,:channels,:,:],
            gt_images=inputs,
            vq_loss=vq_loss,
        )
    
        # always log the first batch and variable amount of timesteps up to 10
        if batch_idx % self.train_log_interval == 0 and self.wandb:
            if reconstructions.shape[0] > 10:
                log_amount = 10
            else:
                log_amount = reconstructions.shape[0]

            # Log input against prediction
            log_images(
                reconstructions[:log_amount],
                inputs[:log_amount],
                log_key="input (left) vs. reconstruction (right)",
                captions=""
            )


        self.log_dict(loss_dict, sync_dist=True, prog_bar=True,
                       batch_size=bs)

        return loss_dict["loss"]

    def on_train_epoch_end(self):
        # gc.collect()
        # torch.cuda.empty_cache()
        return {}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):

        all_center_shapes, label = batch  # TODO this has one dimension too much rn
        self.curr_device = all_center_shapes.device
        bs = all_center_shapes.shape[0]
        channels = all_center_shapes.shape[1]

        out = self.forward(all_center_shapes)
        reconstructions=out[0]
        inputs = all_center_shapes
        vq_loss=out[2]

        loss_dict = self.model.loss_function(
            reconstructions=reconstructions[:,:channels,:,:],
            gt_images=inputs,
            vq_loss=vq_loss,
        )

        self.log_dict({"val_loss": loss_dict["loss"]}, sync_dist=True, prog_bar=True)
        return loss_dict["loss"]

    def on_validation_end(self) -> None:
        # if self.wandb:
        #     self.sample_images()
        # gc.collect()
        # torch.cuda.empty_cache()
        return {}
    
    def configure_optimizers(self):

        optims = []
        scheds = []

        param_group_1 = {'params': self.model.parameters(), 'lr': self.lr}
        param_groups = [param_group_1]

        if not self.weight_decay:
            optimizer = optim.AdamW(
                param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            # learning rates should be explicitly specified in the param_groups
            optimizer = optim.Adam(param_groups)
        optims.append(optimizer)
        
        try:
            if self.scheduler_gamma is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.scheduler_gamma)
                scheds.append(scheduler)

                return optims, scheds
        except:
            pass
        return optims


class VectorGPTExperimentv2(pl.LightningModule):
    def __init__(self,
                 vector_gpt_model: VectorGPTv2,
                 input_mode: str = "layer",
                 lr: float = 0.0003,
                 stroke_lr: float = None,
                 weight_decay: float = 0.0,
                 scheduler_gamma: float = 0.99,
                 train_log_interval: int = 250,
                 manual_seed: int = 42,
                 wandb: bool = True,
                 **kwargs) -> None:
        super(VectorGPTExperimentv2, self).__init__()

        assert input_mode in ["absolute_layer", "centered_layer", "absolute_merged"], "please choose valid input mode in the experiment settings"
        self.model = vector_gpt_model
        self.input_mode = input_mode
        self.lr = lr
        self.stroke_lr = stroke_lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.train_log_interval = train_log_interval
        self.manual_seed = manual_seed
        self.curr_device = None
        self.wandb = wandb

    def forward(self, input_images: Tensor, positions: Tensor, **kwargs) -> Tensor:
        return self.model(input_images, positions, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        input_absolute_shape_layers, input_centered_shape_layers, input_merged_images, stop_signals, captions, target_centered_shape_layers, positions, gt_positions = batch
        self.curr_device = input_absolute_shape_layers.device
        bs = input_absolute_shape_layers.shape[0]

        if self.input_mode == "absolute_layer":
            predicted_shapes, stop_preds, _, pos_preds = self.forward(input_absolute_shape_layers, positions, drop_alpha_channel=False)
        if self.input_mode == "centered_layer":
            predicted_shapes, stop_preds, _, pos_preds = self.forward(input_centered_shape_layers, positions, drop_alpha_channel=False)
        elif self.input_mode == "absolute_merged":
            predicted_shapes, stop_preds, _, pos_preds = self.forward(input_merged_images, positions, drop_alpha_channel=False)

        train_loss, recons_loss, stop_prediction_loss, position_loss = self.model.loss_function(
            gt_shape_layers=target_centered_shape_layers,
            pred_images=predicted_shapes,
            gt_stop_signals=stop_signals,
            stop_signals=stop_preds,
            gt_merged_targets = None,
            merged_preds = None,
            position_predictions = pos_preds,
            gt_positions = gt_positions,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
            log_loss = batch_idx % self.train_log_interval == 0 and self.wandb
        )

        # always log the first batch and variable amount of timesteps up to 10
        if batch_idx % self.train_log_interval == 0 and self.wandb:
            if predicted_shapes[0].shape[0] > 10:
                log_amount = 10
                stop_idx = len(stop_signals[0][stop_signals[0]==0])
                start_idx = torch.randint(0, stop_idx - log_amount, (1,)).item()
            else:
                log_amount = predicted_shapes[0].shape[0]
                start_idx = 0

            # Log input against prediction
            if self.input_mode == "absolute_layer":
                log_images(
                    predicted_shapes[0][start_idx:start_idx+log_amount],
                    input_absolute_shape_layers[0][start_idx:start_idx+log_amount],
                    log_key="input (left) vs. prediction (right)",
                    captions=captions[0] + f" from T={start_idx} to T={start_idx+log_amount}"
                )
            elif self.input_mode == "centered_layer":
                log_images(
                    predicted_shapes[0][start_idx:start_idx+log_amount],
                    input_centered_shape_layers[0][start_idx:start_idx+log_amount],
                    log_key="input (left) vs. prediction (right)",
                    captions=captions[0] + f" from T={start_idx} to T={start_idx+log_amount}"
                )
            elif self.input_mode == "absolute_merged":
                log_images(
                    predicted_shapes[0][start_idx:start_idx+log_amount],
                    input_merged_images[0][start_idx:start_idx+log_amount],
                    log_key="input (left) vs. prediction (right)",
                    captions=captions[0] + f" from T={start_idx} to T={start_idx+log_amount}"
                )

            # log shape prediction against target
            # if merged_preds is not None:  # merged_preds is not None if loss mode is "merged"
            #     log_images(
            #         merged_preds[0][:log_amount],
            #         merged_target[0][:log_amount],
            #         log_key="training merged predictions",
            #         captions=captions[0]
            #     )
            
            # always log the pred shapes and target shapes
            log_images(
                predicted_shapes[0][start_idx:start_idx+log_amount],
                target_centered_shape_layers[0][start_idx:start_idx+log_amount],
                log_key="training predictions",
                captions=captions[0] + f" from T={start_idx} to T={start_idx+log_amount}"
            )


        self.log_dict({"train_loss": train_loss, 
                       "train_recons_loss": recons_loss,
                       "train_stop_prediction_loss": stop_prediction_loss,
                       "position_loss":position_loss}, sync_dist=True, prog_bar=True,
                       batch_size=bs)

        return train_loss

    def on_train_epoch_end(self):
        # gc.collect()
        # torch.cuda.empty_cache()
        return {}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):

        input_absolute_shape_layers, input_centered_shape_layers, input_merged_images, stop_signals, captions, target_centered_shape_layers, positions, gt_positions = batch
        self.curr_device = input_absolute_shape_layers.device

        if self.input_mode == "absolute_layer":
            predicted_shapes, stop_preds, _, pos_preds = self.forward(input_absolute_shape_layers, positions)
        elif self.input_mode == "centered_layer":
            predicted_shapes, stop_preds, _, pos_preds = self.forward(input_centered_shape_layers, positions)
        elif self.input_mode == "absolute_merged":
            predicted_shapes, stop_preds, _, pos_preds = self.forward(input_merged_images, positions)

        val_loss, _, _, position_loss = self.model.loss_function(
            gt_shape_layers=target_centered_shape_layers,
            pred_images=predicted_shapes,
            gt_stop_signals=stop_signals,
            stop_signals=stop_preds,
            gt_merged_targets = None,
            merged_preds = None,
            position_predictions = pos_preds,
            gt_positions = gt_positions,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )


        self.log_dict({"val_loss": val_loss}, sync_dist=True, prog_bar=True)
        return val_loss

    def on_validation_end(self) -> None:
        if self.wandb:
            self.sample_images()
        # gc.collect()
        # torch.cuda.empty_cache()
        return {}
    
    def sample_images(self, num_of_samples = 2):
        input_absolute_shape_layers, input_centered_shape_layers, input_merged_images, stop_signals, captions, target_centered_shape_layers, positions, gt_positions = next(iter(self.trainer.datamodule.val_dataloader()))
        input_absolute_shape_layers = input_absolute_shape_layers[:num_of_samples].to(self.curr_device)
        input_centered_shape_layers = input_centered_shape_layers[:num_of_samples].to(self.curr_device)
        input_merged_images = input_merged_images[:num_of_samples].to(self.curr_device)[:, :, :3, :, :]
        target_centered_shape_layers = target_centered_shape_layers[:num_of_samples].to(self.curr_device)[:, :, :3, :, :]
        positions = positions[:num_of_samples].to(self.curr_device)

        with torch.no_grad():
            if self.input_mode == "absolute_layer":
                predicted_shapes, _, _, _ = self.forward(input_absolute_shape_layers, positions, drop_alpha_channel = True, verbose = True)
            elif self.input_mode == "centered_layer":
                predicted_shapes, _, _, _ = self.forward(input_centered_shape_layers, positions, drop_alpha_channel = True, verbose = True)
            elif self.input_mode == "absolute_merged":
                predicted_shapes, _, _, _ = self.forward(input_merged_images, positions, drop_alpha_channel = True, verbose = True)

            # make sure there are no small negative numbers for rendering
            dummy = torch.nn.ReLU()
            predicted_shapes = dummy(predicted_shapes)
        
        log_images(predicted_shapes[0], target_centered_shape_layers[0], log_key="val_preds", captions=captions[0])
    
    def configure_optimizers(self):

        optims = []
        scheds = []

        param_groups = []

        if self.stroke_lr is not None:
            print(f"[INFO] using separate stroke LR of {self.stroke_lr} instead of {self.lr}")
            stroke_params = []
            other_params = []
            # Separate parameters for the 'stroke_predictor' and other model parameters
            for name, param in self.model.named_parameters():
                if("stroke_predictor" in name):
                    stroke_params.append(param)
                else:
                    other_params.append(param)

            # Set different learning rates for different parameter groups
            param_group_1 = {'params': other_params, 'lr': self.lr}
            param_group_2 = {'params': stroke_params, 'lr': self.stroke_lr}

            param_groups = [param_group_1, param_group_2]
        else:
            param_group_1 = {'params': self.model.parameters(), 'lr': self.lr}
            param_groups = [param_group_1]

        if not self.weight_decay:
            optimizer = optim.AdamW(
                param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            # learning rates should be explicitly specified in the param_groups
            optimizer = optim.Adam(param_groups)
        optims.append(optimizer)
        
        try:
            if self.scheduler_gamma is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.scheduler_gamma)
                scheds.append(scheduler)

                return optims, scheds
        except:
            pass
        return optims

class VectorGPTExperiment(pl.LightningModule):
    def __init__(self,
                 vector_gpt_model: VectorGPT,
                 input_mode: str = "layer",
                 lr: float = 0.0003,
                 stroke_lr: float = None,
                 weight_decay: float = 0.0,
                 scheduler_gamma: float = 0.99,
                 train_log_interval: int = 250,
                 manual_seed: int = 42,
                 wandb: bool = True,
                 **kwargs) -> None:
        super(VectorGPTExperiment, self).__init__()

        assert input_mode in ["layer", "merged"], "please choose valid input mode in the experiment settings"
        self.model = vector_gpt_model
        self.input_mode = input_mode
        self.lr = lr
        self.stroke_lr = stroke_lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.train_log_interval = train_log_interval
        self.manual_seed = manual_seed
        self.curr_device = None
        self.wandb = wandb

    def forward(self, input_shape_layers: Tensor, **kwargs) -> Tensor:
        return self.model(input_shape_layers, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        input_shape_layers, target_shape_layers, stop_signals, captions, merged_target, merged_input = batch
        self.curr_device = input_shape_layers.device
        bs = input_shape_layers.shape[0]

        if self.input_mode == "layer":
            predicted_shapes, stop_preds, merged_preds = self.forward(input_shape_layers, drop_alpha_channel=False)  # TODO was True
        elif self.input_mode == "merged":
            predicted_shapes, stop_preds, merged_preds = self.forward(merged_input, drop_alpha_channel=False)  # TODO was True

        train_loss, recons_loss, stop_prediction_loss = self.model.loss_function(
            gt_shape_layers=target_shape_layers,
            pred_images=predicted_shapes,
            gt_stop_signals=stop_signals,
            stop_signals=stop_preds,
            gt_merged_targets = merged_target,
            merged_preds = merged_preds,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
            log_loss = batch_idx % self.train_log_interval == 0 and self.wandb
        )

        # always log the first batch and variable amount of timesteps up to 10
        if batch_idx % self.train_log_interval == 0 and self.wandb:
            if predicted_shapes[0].shape[0] > 10:
                log_amount = 10
            else:
                log_amount = predicted_shapes[0].shape[0]

            # Log input against prediction
            if self.input_mode == "layer":
                log_images(
                    predicted_shapes[0][:log_amount],
                    input_shape_layers[0][:log_amount],
                    log_key="input (left) vs. prediction (right)",
                    captions=captions[0]
                )
            elif self.input_mode == "merged":
                log_images(
                    predicted_shapes[0][:log_amount],
                    merged_input[0][:log_amount],
                    log_key="input (left) vs. prediction (right)",
                    captions=captions[0]
                )

            # log shape prediction against target
            if merged_preds is not None:  # merged_preds is not None if loss mode is "merged"
                log_images(
                    merged_preds[0][:log_amount],
                    merged_target[0][:log_amount],
                    log_key="training merged predictions",
                    captions=captions[0]
                )
            
            # always log the pred shapes and target shapes
            log_images(
                predicted_shapes[0][:log_amount],
                target_shape_layers[0][:log_amount],
                log_key="training predictions",
                captions=captions[0]
            )


        self.log_dict({"train_loss": train_loss, 
                       "train_recons_loss": recons_loss,
                       "train_stop_prediction_loss": stop_prediction_loss}, sync_dist=True, prog_bar=True,
                       batch_size=bs)

        return train_loss

    def on_train_epoch_end(self):
        # gc.collect()
        # torch.cuda.empty_cache()
        return {}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):

        input_shape_layers, gt_shape_layers, stop_signals, captions, merged_target, merged_input = batch
        self.curr_device = input_shape_layers.device

        if self.input_mode == "layer":
            predicted_shapes, stop_preds, merged_preds = self.forward(input_shape_layers)
        elif self.input_mode == "merged":
            predicted_shapes, stop_preds, merged_preds = self.forward(merged_input)

        val_loss, _, _ = self.model.loss_function(
            gt_shape_layers=gt_shape_layers,
            pred_images=predicted_shapes,
            gt_stop_signals=stop_signals,
            stop_signals=stop_preds,
            gt_merged_targets = merged_target,
            merged_preds = merged_preds,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )


        self.log_dict({"val_loss": val_loss}, sync_dist=True, prog_bar=True)
        return val_loss

    def on_validation_end(self) -> None:
        if self.wandb:
            self.sample_images()
        # gc.collect()
        # torch.cuda.empty_cache()
        return {}
    
    def sample_images(self, num_of_samples=10):
        full_images, shape_layers, stop_signals, captions, merged_target, merged_input = next(iter(self.trainer.datamodule.val_dataloader()))
        test_input = full_images[:num_of_samples].to(self.curr_device)
        test_targets = shape_layers[:num_of_samples].to(self.curr_device)[:, :, :3, :, :]
        merged_input = merged_input[:num_of_samples].to(self.curr_device)[:, :, :3, :, :]

        with torch.no_grad():
            # test_input, test_label = batch
            if self.input_mode == "layer":
                shape_preds, stop_preds, merged_preds = self.model.forward(test_input, drop_alpha_channel=True, verbose=True)
            elif self.input_mode == "merged":
                shape_preds, stop_preds, merged_preds = self.model.forward(merged_input, drop_alpha_channel=True, verbose=True)
        # make sure there are no small negative numbers for rendering
        dummy = torch.nn.ReLU()
        shape_preds = dummy(shape_preds)
        
        log_images(shape_preds[0], test_targets[0], log_key="val_preds", captions=captions[0])
    
    def configure_optimizers(self):

        optims = []
        scheds = []

        param_groups = []

        if self.stroke_lr is not None:
            print(f"[INFO] using separate stroke LR of {self.stroke_lr} instead of {self.lr}")
            stroke_params = []
            other_params = []
            # Separate parameters for the 'stroke_predictor' and other model parameters
            for name, param in self.model.named_parameters():
                if("stroke_predictor" in name):
                    stroke_params.append(param)
                else:
                    other_params.append(param)

            # Set different learning rates for different parameter groups
            param_group_1 = {'params': other_params, 'lr': self.lr}
            param_group_2 = {'params': stroke_params, 'lr': self.stroke_lr}

            param_groups = [param_group_1, param_group_2]
        else:
            param_group_1 = {'params': self.model.parameters(), 'lr': self.lr}
            param_groups = [param_group_1]

        if not self.weight_decay:
            optimizer = optim.AdamW(
                param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            # learning rates should be explicitly specified in the param_groups
            optimizer = optim.Adam(param_groups)
        optims.append(optimizer)
        
        try:
            if self.scheduler_gamma is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.scheduler_gamma)
                scheds.append(scheduler)

                return optims, scheds
        except:
            pass
        return optims


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.beta_scale = 2.0 # introduced with Im2Vec
        self.lr = params["LR"]

        if "offset_LR" in params and self.model.decoder.offset_mode == "learnable":
            self.offset_LR = params["offset_LR"]
        else:
            self.offset_LR = None

        if "offset_warmup" in params and self.model.decoder.offset_mode in ["learnable", "optimizable"]:
            self.offset_warmup = params["offset_warmup"]
        else:
            self.offset_warmup = False
        
        if "log_fid" in self.params.keys():
            self.log_fid = True if self.params["log_fid"] else False
            if self.log_fid:
                self.fid = FrechetInceptionDistance(feature=768, reset_real_features=False, normalize=True)
        else:
            self.log_fid = False
            print("Not logging FID score. To enable, add 'log_fid: True' to the 'exp_params' of the config .yaml file")

        if "clip_sim_model" in self.params.keys() and "clip_prompt_suffix" in self.params.keys():
                self.log_clip_sim = True if self.params["clip_sim_model"] in ['openai/clip-vit-base-patch16', 'openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14-336', 'openai/clip-vit-large-patch14'] else False
                if self.log_clip_sim:
                    self.clip_prompt_suffix = self.params["clip_prompt_suffix"]
                    self.clip_sim = CLIPScore(model_name_or_path=self.params["clip_sim_model"])
                else:
                    self.log_clip_sim = False
                    print(f"""Not logging CLIP similarity score with: {self.params["clip_sim_model"]}. To enable, add one of ['openai/clip-vit-base-patch16', 'openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14-336', 'openai/clip-vit-large-patch14']
                           to the 'clip_sim_model' parameter in 'exp_params' of the config .yaml file""")

        else:
            self.log_clip_sim = False
            print("Not logging CLIP similarity score. To enable, add 'clip_sim_model' to the 'exp_params' of the config .yaml file with one of ['openai/clip-vit-base-patch16', 'openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14-336', 'openai/clip-vit-large-patch14'] and add 'clip_prompt_suffix' also.")

        
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
    
    def on_train_epoch_start(self):
        if self.offset_warmup:
            for name, param in self.model.named_parameters():
                if "decoder.offset" in name or "encoder." in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        # regularely log training reconstructions
        if batch_idx % self.params["train_log_interval"] == 0:
            results = self.forward(real_img, labels = labels, log_path_length=True)
            if results[0].shape[0] > 10:
                log_amount = 10
            else:
                log_amount = results[0].shape[0]
            log_images(results[0][:log_amount], real_img[:log_amount], log_key="training")
            train_loss = self.model.loss_function(*results,
                                            M_N = self.params['kld_weight'],
                                            optimizer_idx=optimizer_idx,
                                            batch_idx = batch_idx,
                                            log_loss_images = True)
        else:
            results = self.forward(real_img, labels = labels)
            train_loss = self.model.loss_function(*results,
                                                M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                                optimizer_idx=optimizer_idx,
                                                batch_idx = batch_idx)

        self.log_dict(train_loss, sync_dist=True, prog_bar=True)

        # Custom processing for Im2Vec
        if(isinstance(self.model, VectorVAEnLayers)):
            path = random.randint(7, 25)
            if self.params['resample_circle_segments']:
                self.model.redo_features(path)

        return train_loss['loss']
    
    def on_train_epoch_end(self):
        if self.offset_warmup:
            for name, param in self.model.named_parameters():
                param.requires_grad = True
        if isinstance(self.model, VectorVAEnLayers):
            if self.current_epoch % 25 ==0:
                new_beta = self.model.beta * self.beta_scale
                self.model.beta = min(new_beta, 4)
        gc.collect()
        torch.cuda.empty_cache()
        return {}

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        print("Entering validation step.")
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val for key, val in val_loss.items()}, sync_dist=True, prog_bar=True)

        
    def on_validation_end(self) -> None:
        print("Entering on_validation_end.")
        with torch.no_grad():
            self.sample_images()

        gc.collect()
        torch.cuda.empty_cache()
        
    def sample_images(self):
        print("Sampling images for wandb.")
        # Get sample reconstruction image            
        if(self.log_clip_sim or self.log_fid):
            num_of_samples = 100
        else:
            num_of_samples = 10
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input[:num_of_samples].to(self.curr_device)
        test_label = test_label[:num_of_samples].to(self.curr_device)

        with torch.no_grad():
            # test_input, test_label = batch
            recons = self.model.generate(test_input, labels = test_label, verbose=True)
        
        # make sure there are no small negative numbers for rendering
        dummy = torch.nn.ReLU()
        recons = dummy(recons)
        
        log_images(recons[:5], test_input[:5], log_key="val_recons")

        # if(self.logger.save_dir is not None):
        #     vutils.save_image(recons.data[:10],
        #                     os.path.join(self.logger.save_dir , 
        #                                 "Reconstructions", 
        #                                 f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                     normalize=True,
        #                     nrow=5)

        try:
            samples = self.model.sample(num_of_samples,
                                        self.curr_device,
                                        labels = test_label)
            samples = dummy(samples[:,:3,:,:]) # drop the alpha channel for metric calculation
            log_images(samples[:5], samples[5:10], log_key="samples")
            # if(self.logger.save_dir is not None):
            #     vutils.save_image(samples.cpu().data,
            #                     os.path.join(self.logger.save_dir , 
            #                                 "Samples",      
            #                                 f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
            #                     normalize=True,
            #                     nrow=5)
            # Log FID score
            if self.log_fid:
                self.fid.update(test_input, real = True)

                # log reconstruction fid
                self.fid.update(recons, real = False)
                fid_recon_score = self.fid.compute()

                # log sample fid
                self.fid.update(samples, real = False)
                fid_sample_score = self.fid.compute()

                self.fid.reset()

                self.logger.log_metrics({"val_recons_FID" : fid_recon_score, "val_sample_FID" : fid_sample_score})#, sync_dist=True ,prog_bar=True)

            if self.log_clip_sim:
                
                _label_translate_dict = self.trainer.datamodule.val_dataset._int_to_label
                # was used to test VRAM usage
                # _max_clip_calculations = 100
                
                self.clip_sim.update(recons, [_label_translate_dict[label]+self.clip_prompt_suffix for label in test_label.cpu().numpy()])
                clip_sim_recon_score = self.clip_sim.compute()
                self.clip_sim.reset()

                self.clip_sim.update(samples, [_label_translate_dict[label]+self.clip_prompt_suffix for label in test_label.cpu().numpy()[:num_of_samples]])
                clip_sim_sample_score = self.clip_sim.compute()
                self.clip_sim.reset()

                self.logger.log_metrics({"val_recons_CLIP_sim" : clip_sim_recon_score, "val_sample_CLIP_sim" : clip_sim_sample_score})#, sync_dist=True ,prog_bar=True)
                
        except Exception as e:
            print(f"[ERROR] at sampling")
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        param_groups = []

        if(self.offset_LR is not None):
            offset_params = []
            other_params = []
            # Separate parameters for the 'offset' layer and other model parameters
            for name, param in self.model.named_parameters():
                if(name == "decoder.offset.bias" or name == "decoder.offset.weight"):
                    offset_params.append(param)
                else:
                    other_params.append(param)

            # Set different learning rates for different parameter groups
            param_group_1 = {'params': other_params, 'lr': self.lr}
            param_group_2 = {'params': offset_params, 'lr': self.offset_LR}

            param_groups = [param_group_1, param_group_2]
        else:
            param_group_1 = {'params': self.model.parameters(), 'lr': self.lr}
            param_groups = [param_group_1]


        if(self.params["weight_decay"] is not None):
            optimizer = optim.AdamW(param_groups,
                                lr=self.lr,
                                weight_decay=self.params['weight_decay'])
        else:
            optimizer = optim.Adam(param_groups,
                                lr=self.lr)
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            pass
        return optims
