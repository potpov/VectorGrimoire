import gc
import os
import math
import random
import torch
from torch import Tensor
from torch import optim
from models import BaseVAE, VectorVAEnLayers, VectorGPT
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from utils import log_images
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore


class VectorGPTExperiment(pl.LightningModule):
    def __init__(self,
                 vector_gpt_model: VectorGPT,
                 lr: float = 0.0003,
                 weight_decay: float = 0.0,
                 scheduler_gamma: float = 0.99,
                 train_log_interval: int = 250,
                 manual_seed: int = 42,
                 wandb: bool = True,
                 **kwargs) -> None:
        super(VectorGPTExperiment, self).__init__()

        self.model = vector_gpt_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.train_log_interval = train_log_interval
        self.manual_seed = manual_seed
        self.curr_device = None
        self.wandb = wandb

    def forward(self, input_shape_layers: Tensor, **kwargs) -> Tensor:
        return self.model(input_shape_layers, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        input_shape_layers, target_shape_layers, stop_signals, captions = batch
        self.curr_device = input_shape_layers.device

        # regularely log training reconstructions
        # if(batch_idx % self.train_log_interval == 0):
        #     predicted_shapes, stop_preds = self.forward(full_images)
        #     # if(results[0].shape[0] > 10):
        #     #     log_amount = 10
        #     # else:
        #     #     log_amount = results[0].shape[0]
        #     # log_images(results[0][:log_amount], real_img[:log_amount], log_key="training")
        #     train_loss = self.model.loss_function(*results,
        #                                     M_N = self.params['kld_weight'],
        #                                     optimizer_idx=optimizer_idx,
        #                                     batch_idx = batch_idx,
        #                                     log_loss_images = True)
        # else:
        predicted_shapes, stop_preds = self.forward(input_shape_layers, drop_alpha_channel=False)  # TODO was True
        train_loss, recons_loss, stop_prediction_loss = self.model.loss_function(
            gt_shape_layers=target_shape_layers,
            pred_images=predicted_shapes,
            gt_stop_signals=stop_signals,
            stop_signals=stop_preds,
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
            log_images(
                predicted_shapes[0][:log_amount],
                target_shape_layers[0][:log_amount],
                log_key="training predctions",
                captions=captions[0]
            )

        self.log_dict({"train_loss": train_loss, 
                       "train_recons_loss": recons_loss,
                       "train_stop_prediction_loss": stop_prediction_loss}, sync_dist=True, prog_bar=True)

        return train_loss
    
    def on_train_epoch_end(self):
        gc.collect()
        torch.cuda.empty_cache()
        return {}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        full_images, shape_layers, stop_signals, captions = batch
        self.curr_device = full_images.device

        predicted_shapes, stop_preds = self.forward(full_images)
        val_loss, _, _ = self.model.loss_function(
            gt_shape_layers=shape_layers,
            pred_images=predicted_shapes,
            gt_stop_signals=stop_signals,
            stop_signals=stop_preds,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )

        self.log_dict({"val_loss": val_loss}, sync_dist=True, prog_bar=True)

        return val_loss

    def on_validation_end(self) -> None:
        if(self.wandb):
            self.sample_images()
        gc.collect()
        torch.cuda.empty_cache()
        return {}
    
    def sample_images(self, num_of_samples=10):
        full_images, shape_layers, stop_signals, captions = next(iter(self.trainer.datamodule.val_dataloader()))
        test_input = full_images[:num_of_samples].to(self.curr_device)
        test_targets = shape_layers[:num_of_samples].to(self.curr_device)[:, :, :3, :, :]

        with torch.no_grad():
            # test_input, test_label = batch
            shape_preds, stop_preds = self.model.forward(test_input, drop_alpha_channel=True, verbose=True)
        
        # make sure there are no small negative numbers for rendering
        dummy = torch.nn.ReLU()
        shape_preds = dummy(shape_preds)
        
        log_images(shape_preds[0], test_targets[0], log_key="val_preds", captions=captions[0])
    
    def configure_optimizers(self):

        optims = []
        scheds = []

        param_groups = []

        param_group_1 = {'params': self.model.parameters(), 'lr': self.lr}
        param_groups = [param_group_1]

        if not self.weight_decay:
            optimizer = optim.AdamW(
                param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = optim.Adam(param_groups, lr=self.lr)
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
