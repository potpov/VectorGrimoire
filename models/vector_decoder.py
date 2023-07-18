import random

import torch
from torch import Tensor
import wandb
from thesis.models import BaseVAE
from torch import nn
from torch.nn import functional as F
from PIL import Image
from typing import List

from utils import fig2data, make_tensor
import pydiffvg
import math
import numpy as np
import kornia
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms

dsample = kornia.transform.PyrDown()


# import os
# import psutil
# process = psutil.Process(os.getpid())


# code taken from Im2Vec and adapted
class VectorDecoder(BaseVAE):


    def __init__(self,
                 latent_dim: int,
                 loss_fn: str = 'MSE',
                 paths: int = 4,
                 wandb_logging = None,
                 **kwargs) -> None:
        super(VectorDecoder, self).__init__()

        self.wandb_logging = wandb_logging
        self.latent_dim = latent_dim
        self.beta = kwargs['beta']
        self.other_losses_weight = 0
        self.reparametrize_ = False
        if 'other_losses_weight' in kwargs.keys():
            self.other_losses_weight = kwargs['other_losses_weight']
        if 'reparametrize' in kwargs.keys():
            self.reparametrize_ = kwargs['reparametrize']

        self.curves = paths
        # self.in_channels = in_channels
        self.scale_factor = kwargs['scale_factor']
        self.learn_sampling = kwargs['learn_sampling']
        self.only_auxillary_training = kwargs['only_auxillary_training']
        self.memory_leak_training = kwargs['memory_leak_training']

        self.memory_leak_epochs = 105
        if 'memory_leak_epochs' in kwargs.keys():
            self.memory_leak_epochs = kwargs['memory_leak_epochs']

        if loss_fn == 'BCE':
            self.loss_fn = F.binary_cross_entropy_with_logits
        else:
            self.loss_fn = F.mse_loss

        self.circle_rad = kwargs['radius']
        self.number_of_points = self.curves * 3

        sample_rate = 1
        angles = torch.arange(0, self.number_of_points, dtype=torch.float32) *6.28319/ self.number_of_points
        id = self.sample_circle(self.circle_rad, angles, sample_rate)
        base_control_features = torch.tensor([[1,0],[0,1],[0,1]], dtype=torch.float32)
        self.id = id[:,:]
        self.angles = angles
        self.register_buffer('base_control_features', base_control_features)
        self.deformation_range = 6.28319/ 4 # TODO why division by 4

        def get_computational_unit(in_channels, out_channels, unit):
            if unit=='conv':
                return nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular', stride=1, dilation=1) # TODO padding was 2 before
            else:
                return nn.Linear(in_channels, out_channels)
            # Build Decoder

        unit='conv'
        if unit=='conv':
            self.decode_transform = lambda x: x.permute(0, 2, 1)
        else:
            self.decode_transform = lambda x: x
        num_one_hot = base_control_features.shape[1]
        fused_latent_dim = latent_dim + num_one_hot+ (sample_rate*2)
        self.decoder_input = get_computational_unit(fused_latent_dim, fused_latent_dim*2, unit)

        self.point_predictor = nn.ModuleList([
            get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
            get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
            get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
            get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
            get_computational_unit(fused_latent_dim*2, 2, unit),
            # nn.Sigmoid()  # bound spatial extent
        ])
        if self.learn_sampling:
            self.sample_deformation = nn.Sequential(
                get_computational_unit(latent_dim + 2+ (sample_rate*2), latent_dim*2, unit),
                nn.ReLU(),
                get_computational_unit(latent_dim * 2, latent_dim * 2, unit),
                nn.ReLU(),
                get_computational_unit(latent_dim*2, 1, unit),
            )
        self.aux_network = nn.Sequential(
            get_computational_unit(latent_dim, latent_dim*2, 'mlp'),
            nn.LeakyReLU(),
            get_computational_unit(latent_dim * 2, latent_dim * 2, 'mlp'),
            nn.LeakyReLU(),
            get_computational_unit(latent_dim * 2, latent_dim * 2, 'mlp'),
            nn.LeakyReLU(),
            get_computational_unit(latent_dim*2, 3, 'mlp'),
        )
        self.latent_lossvpath = {}
        self.save_lossvspath = False
        if self.only_auxillary_training:
            self.save_lossvspath = True
            for name, param in self.named_parameters():
                if 'aux_network' in name:
                    print(name)
                    param.requires_grad =True
                else:
                    param.requires_grad =False
        # self.lpips = VGGPerceptualLoss(False)

    def redo_features(self, n):
        self.curves = n
        self.number_of_points = self.curves * 3
        # aranges the points between 0 and 2pi
        self.angles = (torch.arange(0, self.number_of_points, dtype=torch.float32) *6.28319/ self.number_of_points)

        id = self.sample_circle(self.circle_rad, self.angles, 1)
        self.id = id[:,:]

    def control_polygon_distance(self, all_points):
        def distance(vec1, vec2):
            return ((vec1-vec2)**2).mean()

        loss =0
        for idx in range(self.number_of_points):
            c_0 = all_points[:, idx - 1, :]
            c_1 = all_points[:, idx, :]
            loss = loss + distance(c_0, c_1)
        return loss

    def sample_circle(self, r, angles, sample_rate=10):
        pos = []
        for i in range(1, sample_rate+1):
            x = (torch.cos(angles*(sample_rate/i)) * r)# + r
            y = (torch.sin(angles*(sample_rate/i)) * r)# + r
            pos.append(x)
            pos.append(y)
        return torch.stack(pos, dim=-1)

    def raster(self, all_points, color=[0,0,0, 1], verbose=False, white_background=True, **kwargs):
        assert len(color) == 4
        # print('1:', process.memory_info().rss*1e-6)
        render_size = self.imsize
        bs = all_points.shape[0]
        if verbose:
            render_size = render_size*2
        outputs = []
        all_points = all_points*render_size # brings point coordinates from [0,1] back to image scale

        num_ctrl_pts = torch.zeros(self.curves, dtype=torch.int32).to(all_points.device) + 2
        if(isinstance(color, list)):
            color = make_tensor(color, grad=True).to(all_points.device)
        else:
            color.to(all_points.device)
        for k in range(bs):
            # Get point parameters from network
            render = pydiffvg.RenderFunction.apply
            shapes = []
            shape_groups = []
            points = all_points[k].contiguous()#[self.sort_idx[k]] # .cpu()

            # had this issue a few times with the LR finder
            if(torch.isnan(points).any()):
                print(f"[WARNING] Found NaN values in points")
                points = torch.rand(points.shape).to(points.device)* 0.01

            # check if points are all collapsed, would throw DiffVG error..
            if(torch.all(points == 0.0)):
                print(f"[WARNING] Found all points to be at 0.0")
                noise_vec = torch.rand(points.shape).to(points.device)* 0.01
                points = points + noise_vec


            if verbose: # I think this creates this color gradient for easier visual tracing in the rastered image
                np.random.seed(0)
                colors = np.random.rand(self.curves, 4)
                high = np.array((0.565, 0.392, 0.173, 1))
                low = np.array((0.094, 0.310, 0.635, 1))
                diff = (high-low)/(self.curves)
                colors[:, 3] = 1
                for i in range(self.curves):
                    scale = diff*i
                    color = low + scale
                    color[3] = 1
                    color = torch.tensor(color)
                    num_ctrl_pts = torch.zeros(1, dtype=torch.int32) + 2
                    if i*3 + 4 > self.curves * 3:
                        curve_points = torch.stack([points[i*3], points[i*3+1], points[i*3+2], points[0]])
                    else:
                        curve_points = points[i*3:i*3 + 4]
                    path = pydiffvg.Path(
                        num_control_points=num_ctrl_pts, points=curve_points,
                        is_closed=False, stroke_width=torch.tensor(4))
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([i]),
                        fill_color=None,
                        stroke_color=color)
                    shapes.append(path)
                    shape_groups.append(path_group)
                for i in range(self.curves * 3):#from here TODO comment
                    scale = diff*(i//3)
                    color = low + scale
                    color[3] = 1
                    color = torch.tensor(color)
                    if i%3==0:
                        # color = torch.tensor(colors[i//3]) #green
                        shape = pydiffvg.Rect(p_min = points[i]-8,
                                             p_max = points[i]+8)
                        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.curves+i]),
                                                           fill_color=color)
                
                    else:
                        # color = torch.tensor(colors[i//3]) #purple
                        shape = pydiffvg.Circle(radius=torch.tensor(8.0),
                                                 center=points[i])
                        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.curves+i]),
                                                           fill_color=color)
                    shapes.append(shape)
                    shape_groups.append(group)

            else:

                path = pydiffvg.Path(
                    num_control_points=num_ctrl_pts, points=points,
                    is_closed=True)

                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=color,
                    stroke_color=color)
                shape_groups.append(path_group)
            scene_args = pydiffvg.RenderFunction.serialize_scene(render_size, render_size, shapes, shape_groups)
            out = render(render_size,  # width
                         render_size,  # height
                         3,  # num_samples_x
                         3,  # num_samples_y
                         102,  # seed
                         None,
                         *scene_args)
            out = out.permute(2, 0, 1).view(4, render_size, render_size)#[:3]#.mean(0, keepdim=True)
            outputs.append(out)
        output =  torch.stack(outputs).to(all_points.device)

        # map to [-1, 1]
        if white_background:
            alpha = output[:, 3:4, :, :]
            output_white_bg = output[:, :3, :, :]*alpha + (1-alpha)
            output = torch.cat([output_white_bg, alpha], dim=1)
        del num_ctrl_pts#, color
        return output

    def decode(self, z: Tensor, point_predictor=None, verbose=False) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        if point_predictor==None:
            point_predictor = self.point_predictor
        self.id = self.id.to(z.device) # [self.curves * 3, 2], I think this is the x-y position p

        bs = z.shape[0]
        z = z[:, None, :].repeat([1, self.curves *3, 1])
        base_control_features = self.base_control_features[None, :, :].repeat(bs, self.curves, 1 ) # I think this is the control variable c
        z_base = torch.cat([z, base_control_features], dim=-1)
        z_base_transform = self.decode_transform(z_base) # TODO why is this commented out
        if self.learn_sampling:
            self.angles = self.angles.to(z.device)
            angles= self.angles[None, :, None].repeat(bs, 1, 1)
            x = torch.cos(angles)# + r
            y = torch.sin(angles)# + r
            z_angles = torch.cat([z_base, x, y], dim=-1)

            angles_delta = self.sample_deformation(self.decode_transform(z_angles))
            angles_delta = F.tanh(angles_delta/50)*self.deformation_range
            angles_delta = self.decode_transform(angles_delta)

            new_angles = angles + angles_delta
            x = (torch.cos(new_angles) * self.circle_rad)# + r
            y = (torch.sin(new_angles) * self.circle_rad)# + r
            z = torch.cat([z_base, x, y], dim=-1)
        else:
            id = self.id[None, :, :].repeat(bs, 1, 1)
            z = torch.cat([z_base, id], dim=-1) # [bs, self.curves * 3, latent_dim + 2 (c) + 2 (p)]

        all_points = self.decoder_input(self.decode_transform(z))
        for compute_block in point_predictor:
            all_points = F.relu(all_points)
            # all_points = torch.cat([z_base_transform, all_points], dim=1)
            all_points = compute_block(all_points)
        all_points = self.decode_transform(F.sigmoid(all_points/self.scale_factor))
        return all_points

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        if self.reparametrize_:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def forward(self, z: Tensor, **kwargs) -> List[Tensor]:
        all_points = self.decode(z)
        if not self.only_auxillary_training or self.save_lossvspath:
            output = self.raster(all_points, white_background=True)
        else:
            output = torch.zeros([1,3,64,64])
        return  [output, z, -1.0, -1.0]

    def bilinear_downsample(self, tensor, size):
        return torch.nn.functional.interpolate(tensor, size, mode='bilinear')

    def log_losses(self, all_inputs, all_recons):
        all_losses = self.loss_fn(all_recons, all_inputs, reduction='none').mean(dim=1).detach().cpu().numpy()
        cm = plt.get_cmap("Reds")
        # [plt.imsave(f"figures/loss_{i}.png", all_losses[i], cmap="Oranges") for i in range(len(all_losses))]
        # loss_images = [Image.fromarray(cm(all_losses[i]), mode="RGBA") for i in range(len(all_losses))]
        wandb.log({"pyramid_loss":[wandb.Image(cm(image)) for image in all_losses]})

    def log_pyramid_images(self, all_recons, all_real_imgs, log_loss = True):
        log_str = "pyramid_recons"
        assert len(all_recons) == len(all_real_imgs), "reconstruction and input images have different lenghts"

        def get_concat_h(im1, im2):
            dst = Image.new('RGB', (im1.width + im2.width, im1.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width, 0))
            return dst

        with torch.no_grad():
            pyramid_log_dict = {}
            for pyramid_step in range(len(all_recons)):
                input_imgs_t = [transforms.ToPILImage()(all_real_imgs[pyramid_step][i]).convert("RGB") for i in range(len(all_real_imgs[pyramid_step]))]
                recons_imgs_t = [transforms.ToPILImage()(all_recons[pyramid_step][i]).convert("RGB") for i in range(len(all_recons[pyramid_step]))]

                combined_imgs_t = [get_concat_h(input_imgs_t[i], recons_imgs_t[i]) for i in range(len(input_imgs_t))]

                pyramid_log_dict[f"{log_str}_step_{pyramid_step}"] =  [wandb.Image(image) for image in combined_imgs_t]
            
            if(log_loss):
                cm = plt.get_cmap("Reds")
                for pyramid_step in range(len(all_recons)):
                    all_losses_t = self.loss_fn(all_recons[pyramid_step], all_real_imgs[pyramid_step], reduction='none').mean(dim=1).detach().cpu().numpy()
                    pyramid_log_dict[f"{log_str}_step_{pyramid_step}_loss"] =  [wandb.Image(cm(image)) for image in all_losses_t]
                    
        wandb.log(pyramid_log_dict)

    def gaussian_pyramid_loss(self, recons, input, log_loss_images = False):
        recon_loss =self.loss_fn(recons, input, reduction='none').mean(dim=[1,2,3]) #+ self.lpips(recons, input)*0.1
        all_recons = [recons]
        all_inputs = [input]
        for j in range(2,5):
            recons = dsample(recons)
            input = dsample(input)
            if(log_loss_images):
                all_recons.append(recons)
                all_inputs.append(input)
            recon_loss = recon_loss + self.loss_fn(recons, input, reduction='none').mean(dim=[1,2,3])/j
        if(log_loss_images):
            self.log_pyramid_images(all_recons, all_inputs, log_loss=True)
        return recon_loss

    def loss_function(self,
                      recons:Tensor,
                      input:Tensor,
                      mu,
                      log_var,
                      other_losses:int = 0,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        aux_loss = 0
        kld_loss = 0
        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        if not self.only_auxillary_training or self.save_lossvspath:
            if("log_loss_images" in kwargs.keys() and self.wandb_logging):
                recon_loss = self.gaussian_pyramid_loss(recons, input, log_loss_images = kwargs["log_loss_images"])
            else:
                recon_loss = self.gaussian_pyramid_loss(recons, input, log_loss_images = False)
        else:
            recon_loss = torch.zeros([1])
        if self.only_auxillary_training:
            recon_loss_non_reduced = recon_loss[:, None].clone().detach()
            spacing = self.aux_network(mu.clone().detach())
            latents = mu.cpu().numpy()
            num_latents = latents.shape[0]
            if self.save_lossvspath:
                recon_loss_non_reduced_cpu = recon_loss_non_reduced.cpu().numpy()
                keys  = self.latent_lossvpath.keys()
                for i in range(num_latents):
                    if np.array2string(latents[i]) in keys:
                        pair = make_tensor([self.curves, recon_loss_non_reduced_cpu[i, 0], ])[None, :].to(mu.device)
                        self.latent_lossvpath[np.array2string(latents[i])]\
                            = torch.cat([self.latent_lossvpath[np.array2string(latents[i])], pair], dim=0)
                    else:
                        self.latent_lossvpath[np.array2string(latents[i])] = make_tensor([[self.curves, recon_loss_non_reduced_cpu[i, 0]], ]).to(mu.device)
                num = torch.ones_like(spacing[:, 0]) * self.curves
                est_loss = spacing[:,2] + 1/torch.exp(num*spacing[:,0] - spacing[:,1])
                # est_loss = spacing[:, 2] + (spacing[i, 0] / num)

                aux_loss = torch.abs(num*(est_loss - recon_loss_non_reduced)).mean() * 10
            else:
                aux_loss = 0
                for i in range(num_latents):
                    pair = self.latent_lossvpath[np.array2string(latents[i])]
                    est_loss = spacing[i, 2] + 1 / torch.exp(pair[:, 0] * spacing[i, 0] - spacing[i, 1])

                    # est_loss = spacing[i, 2] + (spacing[i, 0] / pair[:, 0])
                    aux_loss = aux_loss + torch.abs(pair[:, 0]*(est_loss - pair[:, 1])).mean()
            loss =  aux_loss
            kld_loss = 0#self.beta*kld_weight * kld_loss
            logs = {'Reconstruction_Loss': recon_loss.mean(), 'KLD': -kld_loss, 'aux_loss': aux_loss}

            logs["loss"] = loss
            return logs
        recon_loss = recon_loss.mean()
        if self.beta>0:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            kld_loss = self.beta*kld_weight * kld_loss
        recon_loss = recon_loss*10
        loss =  recon_loss + kld_loss + other_losses*self.other_losses_weight
        logs = {'Reconstruction_Loss': recon_loss, 'KLD': -kld_loss, 'aux_loss': aux_loss, 'other losses': other_losses*self.other_losses_weight}
        logs["self.beta"] = self.beta
        logs["final_kld_weight"] = self.beta*kld_weight
        logs["loss"] = loss
        if(self.wandb_logging):
            wandb.log(logs)
        return logs

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        all_points = self.decode(z)
        samples = self.raster(all_points, **kwargs)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  self.raster(self.decode(z), verbose=random.choice([True, False]))
 # .type(torch.FloatTensor).to(device)

    def save(self, x, save_dir, name):
        z, log_var = self.encode(x)
        all_points = self.decode(z)
        # print(all_points.std(dim=1))
        # all_points = ((all_points-0.5)*2 + 0.5)*self.imsize
        # if type(self.sort_idx) == type(None):
        #     angles = torch.atan(all_points[:,:,1]/all_points[:,:,0]).detach()
        #     self.sort_idx = torch.argsort(angles, dim=1)
        # Process the batch sequentially
        outputs = []
        for k in range(1):
            # Get point parameters from network
            shapes = []
            shape_groups = []
            points = all_points[k].cpu()#[self.sort_idx[k]]

            color = torch.cat([torch.tensor([0,0,0,1]),])
            num_ctrl_pts = torch.zeros(self.curves, dtype=torch.int32) + 2

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                is_closed=True)

            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=color,
                stroke_color=color)
            shape_groups.append(path_group)
            pydiffvg.save_svg(f"{save_dir}{name}/{name}.svg",
                              self.imsize, self.imsize, shapes, shape_groups)


    def interpolate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(mu.shape[0]):
            z = self.interpolate_vectors(mu[2], mu[i], 10)
            all_points = self.decode(z)
            all_interpolations.append(self.raster(all_points, verbose=kwargs['verbose']))
        return all_interpolations

    def interpolate2D(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        y_axis = self.interpolate_vectors(mu[7], mu[6], 10)
        for i in range(10):
            z = self.interpolate_vectors(y_axis[i], mu[3], 10)
            all_points = self.decode(z)
            all_interpolations.append(self.raster(all_points, verbose=kwargs['verbose']))
        return all_interpolations


    def naive_vector_interpolate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_points = self.decode(mu)
        all_interpolations = []
        for i in range(mu.shape[0]):
            z = self.interpolate_vectors(all_points[2], all_points[i], 10)
            all_interpolations.append(self.raster(z, verbose=kwargs['verbose']))
        return all_interpolations

    def visualize_sampling(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(5,27):
            self.redo_features(i)
            all_points = self.decode(mu)
            all_interpolations.append(self.raster(all_points, verbose=kwargs['verbose']))
        return all_interpolations

    def sampling_error(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        error = []
        figure = plt.figure(figsize=(6, 6))
        bs = x.shape[0]
        for i in range(7,25):
            self.redo_features(i)
            results = self.forward(x)
            recons = results[0][:, :3, :, :]
            input_batch = results[1]

            recon_loss = self.gaussian_pyramid_loss(recons, input_batch)
            print(recon_loss)
            error.append(recon_loss)
        etn = torch.stack(error, dim=1).numpy()
        np.savetxt('sample_error.csv', etn, delimiter=',')
        y = np.arange(7,25)
        for i in range(bs):
            plt.plot(y, etn[i,:], label=str(i+1))
        plt.legend(loc='upper right')
        img = fig2data(figure)
        return img

    def visualize_aux_error(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        bs = mu.shape[0]
        all_spacing = []
        figure = plt.figure(figsize=(6, 6))

        for i in np.arange(7,25):
            spacing = self.aux_network(mu.clone().detach())
            num = torch.ones_like(spacing[:,0])*i
            # est_loss = spacing[:,2] + 1/torch.exp(num*spacing[:,0] + spacing[:,1])
            est_loss =     spacing[:,2] + (spacing[:,0]/num)

            # print(i, spacing[0])
            all_spacing.append(est_loss)
        all_spacing = torch.stack(all_spacing, dim=1).detach().cpu().numpy()
        y = np.arange(7,25)
        for i in range(bs):
            plt.plot(y, all_spacing[i,:], label=str(i+1))
        plt.legend(loc='upper right')
        img = fig2data(figure)
        return img
