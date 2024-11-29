from typing import Tuple, Union
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import wandb
from utils import log_all_images, tensor_to_histogram_image, calculate_global_positions, shapes_to_drawing, svg_string_to_tensor, width_pred_to_local_stroke_width
from models.vsq_heads import MLPVectorHead, CNNVectorHead, VectorHydra
from models.vq_vae import VectorQuantizer
from torchvision.models import ResNet, resnet18
from vector_quantize_pytorch import FSQ
from x_transformers import TransformerWrapper, Decoder
from transformers import BertModel
from svgwrite import Drawing
from einops import rearrange
from kornia.color import rgb_to_lab, rgba_to_rgb, rgb_to_grayscale


class DeconvResNet(nn.Module):
    """
    This class only exists for debugging and validation purposes.
    It is used to validate if everything in the VSQ works when paired with a regular pixel-based prediction head.
    """

    def __init__(self):
        super(DeconvResNet, self).__init__()

        # Define layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = F.sigmoid(self.deconv5(x))  # Using sigmoid for the final layer to scale values between 0 and 1

        return x, {}


class VSQ(nn.Module):
    """
    Vector Shape/Stroke Quantizer.
    Vector quantized pre-training of an autoencoder for SVG primitives.

    Input/Output are shape layers (or patches), no positions. Positions are leraned using the Transformer in Stage II.
    """

    def __init__(self,
                 vector_decoder_model: str = "mlp",
                 quantized_dim: int = 256,
                 codebook_size: int = 512,
                 patch_size: int = 128,
                 image_loss: str | list = "pyramid",
                 num_codes_per_shape: int = 1,
                 vq_method: str = "fsq",
                 fsq_levels: list = [8, 5, 5, 5],
                 num_segments: int = 1,
                 #  geometric_constraint: str = None,
                 alpha: float = 0.0,
                 pred_color: bool = False,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(VSQ, self).__init__()

        assert vector_decoder_model in [
            "mlp", "raster_conv","cnn", "hydra"], \
            "vector_decoder_model must be one of ['mlp', 'raster_conv', 'cnn', 'hydra']"
        # assert geometric_constraint in ["inner_distance", None], f"geometric_constraint must be one of ['inner_distance'], but was {geometric_constraint}"

        self.vector_decoder_model = vector_decoder_model
        self.quantized_dim = quantized_dim
        self.image_loss = image_loss
        self.vq_method = vq_method.lower()

        assert self.vq_method == "fsq", "Please use FSQ."
        self.fsq_levels = fsq_levels
        self.num_segments = num_segments
        self.num_codes_per_shape = num_codes_per_shape
        self.pred_color = pred_color
        self.patch_size = patch_size
        self.dropout = dropout

        if alpha > 0.0:
            self.geometric_constraint = "inner_distance"
            self.alpha = alpha
        else:
            self.geometric_constraint = "None"
            self.alpha = 0.0

        if self.vq_method == "fsq":
            self.codebook_size = np.prod(fsq_levels)
        else:
            self.codebook_size = codebook_size

        self.encoder = resnet18(num_classes=self.quantized_dim * self.num_codes_per_shape)

        if self.vq_method == "vqvae":
            self.quantize_layer = VectorQuantizer(num_embeddings=self.codebook_size,
                                                  embedding_dim=self.quantized_dim,
                                                  beta=0.25)
        elif self.vq_method == "fsq":
            self.quantize_layer = FSQ(levels=self.fsq_levels,
                                      dim=self.quantized_dim)
        elif self.vq_method == "vqtorch":
            raise NotImplementedError("VQVAE with vqtorch not implemented yet.")
        else:
            raise ValueError(f"vq_method must be one of ['vqvae', 'fsq', 'vqtorch'], but is {self.vq_method}")

        self.latent_dim = self.quantized_dim

        if self.vector_decoder_model == "mlp":
             self.decoder = MLPVectorHead(latent_dim=self.quantized_dim * self.num_codes_per_shape,
                                         segments=self.num_segments,
                                         imsize=self.patch_size,
                                         max_stroke_width=20.,
                                         pred_color=self.pred_color,
                                         dropout=self.dropout, )
        elif self.vector_decoder_model == "cnn":
            self.decoder = CNNVectorHead(latent_dim=self.quantized_dim * self.num_codes_per_shape,
                                         segments=self.num_segments,
                                         imsize=self.patch_size,
                                         max_stroke_width=20.,
                                         pred_color=self.pred_color,
                                         )
        elif self.vector_decoder_model == "hydra":
            self.decoder = VectorHydra(
                     latent_dim=self.quantized_dim * self.num_codes_per_shape,
                     segments=self.num_segments,
                     imsize=self.patch_size,
                     max_stroke_width=20.,
            )
        elif self.vector_decoder_model == "raster_conv":
            self.decoder = DeconvResNet()

    def encode(self, input: Tensor, quantize: bool = False):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) latent codes
        """
        result = self.encoder.forward(
            input)  # output from default resnet pytorch is (bs, self.quantized_dim * self.num_codes_per_shape)
        while result.dim() < 4:
            result = result.unsqueeze(-1)
        if self.num_codes_per_shape > 1:
            result = rearrange(result, 'b (c2 c) h w -> b c2 (c h) w', c2=self.quantized_dim)
        # result = self.mapping_layer(result.view(-1, 512 * 4 * 4))
        if quantize:
            result = self.quantize_layer.forward(result)  # this might change the result return type to list
        return result

    def decode(
            self,
            z: Tensor
    ) -> Tensor:
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder.forward(z)

    def decode_from_indices(self, idxs: Tensor) -> Union[Tensor, dict]:
        """
        Maps the given idxs to [reconstructions, input, all_points, vq_loss], all_points are the points of the bezier curves
        :param z: (Tensor) [B x 1]

        """
        if self.vq_method == "fsq":
            codes = self.quantize_layer.indices_to_codes(idxs)
        else:
            raise NotImplementedError("Only FSQ implemented for now.")
        # concat the codes
        if self.num_codes_per_shape > 1:
            b_dim = int(codes.shape[0] / self.num_codes_per_shape)
            codes = codes.view(b_dim, self.num_codes_per_shape, self.quantized_dim).permute(0, 2, 1).unsqueeze(-1)
            codes = rearrange(codes, 'b d (c h) w -> b (d c) h w', c=self.num_codes_per_shape)
            codes = codes.view(b_dim, self.quantized_dim * self.num_codes_per_shape)

        result, logging_dict = self.decode(codes)
        # if self.vector_decoder_model == "mlp":
        #     result = result[0]  # extract only the raster image for now
        return result, logging_dict

    def forward(self, input: Tensor, logging=False, return_visual_attributes=False, only_return_recons=False, **kwargs):
        """
        visual_attribute_dict = {
            "stroke_widths" : all_widths,
            "alphas" : all_alphas,
            "colors": all_colors
        }
        """
        logging_dict = {}
        bs = input.shape[0]
        encoding = self.encode(input, quantize=False)
        vq_logging_dict = {}
        if self.vector_decoder_model in ["mlp", "cnn", "hydra"]:
            # quantize the encoding
            if self.vq_method == "vqvae":
                quantized_inputs, vq_loss, vq_logging_dict = self.quantize_layer.forward(encoding, logging=logging)
            elif self.vq_method == "fsq":
                quantized_inputs, indices = self.quantize_layer.forward(encoding)
                vq_loss = torch.tensor(0.)
                if logging:
                    vq_logging_dict = {
                        "codebook_histogram": wandb.Image(
                            tensor_to_histogram_image(indices.detach().flatten().cpu()),
                            caption="histogram of codebook indices"
                        )
                    }

            # flatten it for MLP digestion
            # quantized_inputs = quantized_inputs.permute(0,2,1,3)
            quantized_inputs = rearrange(quantized_inputs, 'b d (c h) w -> b (d c) h w', c=self.num_codes_per_shape)
            quantized_inputs = quantized_inputs.view(bs, self.quantized_dim * self.num_codes_per_shape)
            # print("quantized_inputs: ", quantized_inputs.shape)
        elif self.vector_decoder_model == "raster_conv":
            quantized_inputs, vq_loss = self.quantize_layer(encoding)

        # re-merge the quantized codes
        # quantized_inputs = rearrange(quantized_inputs, 'b d (c h) w -> b (d c) h w', c=self.num_codes_per_shape)
        # for mlp out is [output, scenes, all_points, all_widths]
        out, decode_logging_dict = self.decode(quantized_inputs)
        logging_dict = {**logging_dict, **decode_logging_dict, **vq_logging_dict}

        if len(out) == 4:
            reconstructions, scenes, all_points, visual_attribute_dict = out
            if only_return_recons:
                return reconstructions
            if return_visual_attributes:
                return [reconstructions, input, all_points, vq_loss, visual_attribute_dict, scenes], logging_dict
            else:
                return [reconstructions, input, all_points, vq_loss, scenes], logging_dict

        reconstructions, (all_paths, all_groups), all_points, visual_attribute_dict, outline, bnw = out
        return [reconstructions, input, all_points, vq_loss, (all_paths, all_groups), outline,
                bnw], logging_dict


    def gaussian_pyramid_loss(
            self,
            recons_images: Tensor,
            gt_images: Tensor,
            down_sample_steps: int = 3,
            log_loss: bool = False,
            pyramid_weights: Tensor = None,
            color_weights: Tensor = None,
            color_masks: Tensor = None
    ):
        """
        Calculates the gaussian pyramid loss between reconstructed images and ground truth images.

        Args:
            - recons_images (Tensor): Reconstructed images in format (-1, c, w, h)
            - gt_images (Tensor): Ground truth images in format (-1, c, w, h)
            - down_sample_steps (int): Number of downsample steps to calculate the loss for. Default: 3
            - filter_mask (Tensor): Keep loss contribution on those pixels. Default: None
            - color_weights (Tensor): Weights for each layer according to color. Default: None
        Returns:
            - recon_loss (Tensor): The gaussian pyramid loss between reconstructed images and ground truth images.
        """
        dsample = kornia.geometry.transform.pyramid.PyrDown()
        timesteps_to_log = 4
        recon_loss = F.mse_loss(recons_images, gt_images, reduction='none')
        L, AB = torch.split(recon_loss, [1, 2], dim=1)

        if not torch.is_tensor(color_weights):  # deactivate color weights if not provided
            color_weights = torch.ones(L.shape[0]).to(L.device)

        if log_loss:
            all_loss_images = []
            all_loss_images.append(self.transform_loss_tensor_to_image(recon_loss[:timesteps_to_log]))

        lab_scale_factor = 50
        ab_loss = sum([AB[i][color_masks[i]].mean() * lab_scale_factor * color_weights[i] for i in range(len(color_masks))]) * pyramid_weights[0]
        l_loss = sum(L.mean((1,2,3))) * pyramid_weights[0]

        recons_loss_contributions = {
            f"AB_pyramid_loss_step_0": ab_loss.detach().cpu().item(),
            f"L_pyramid_loss_step_0": l_loss.detach().cpu().item()
        }

        for j in range(1, 1 + down_sample_steps):
            if j < len(pyramid_weights) and pyramid_weights[j] == 0:
                pyramid_weight = pyramid_weights[j]
            else:
                continue
            recons_images = dsample(recons_images)
            gt_images = dsample(gt_images)

            loss_images = F.mse_loss(recons_images, gt_images, reduction='none')
            L, AB = torch.split(recon_loss, [1, 2], dim=1)
            curr_pyramid_loss_AB = sum(
                [AB[i][color_masks[i]].mean() * lab_scale_factor * color_weights[i] for i in range(len(color_masks))]
            ) * pyramid_weight
            curr_pyramid_loss_L = sum(L.mean((1, 2, 3))) * pyramid_weight

            if log_loss:
                all_loss_images.append(self.transform_loss_tensor_to_image(loss_images[:timesteps_to_log]))

            recons_loss_contributions[f"AB_pyramid_loss_step_{j}"] = curr_pyramid_loss_AB.detach().cpu().item()
            recons_loss_contributions[f"L_pyramid_loss_step_{j}"] = curr_pyramid_loss_L.detach().cpu().item()
            ab_loss = ab_loss + curr_pyramid_loss_AB
            l_loss = l_loss + curr_pyramid_loss_L

        if log_loss:
            log_all_images(
                all_loss_images, log_key="pyramid loss",
                caption=f"Gaussian Pyramid Loss, {down_sample_steps + 1} steps"
            )
            wandb.log(recons_loss_contributions)
        return ab_loss, l_loss, recons_loss_contributions

    def _get_mean_inner_distance(self,
                                 points: Tensor,
                                 use_neighbors_only: bool = False) -> Tensor:
        """
        mean inner distance is defined as the distance between start and end point of each segment of the path

        returns batched mean
        """
        inner_dists = []
        # TODO experiment with quadratic distance here
        for i in range(self.num_segments if use_neighbors_only else self.num_segments + 1):
            if use_neighbors_only:
                inner_dist = torch.cdist(points[:, :, i * 3, :], points[:, :, (i + 1) * 3, :], p=2)
                inner_dists.append(inner_dist.mean())
            else:
                inner_dist = None
                for j in range(self.num_segments + 1):
                    if i != j:
                        ij_dist = torch.cdist(points[:, :, i * 3, :], points[:, :, j * 3, :], p=2)
                        inner_dist = inner_dist + ij_dist if inner_dist is not None else ij_dist
                inner_dists.append(inner_dist.squeeze() / self.num_segments)
        return torch.mean(torch.stack(inner_dists, dim=1), dim=1)

    def _get_inner_distance_penalty(self, points: Tensor):
        """
        input: points, Tensor, (bs, 3*num_segments+1, 2)

        inner distance penalty punishes points to be non-equally distributed.
        it does this by calculating the mean scaled distance between each point and all other points
        """
        inner_penalties = []
        for j in range(self.num_segments + 1):
            inner_dists = []
            for i in range(self.num_segments + 1):
                if i != j:
                    ij_dist = torch.cdist(points[:, :, i * 3, :], points[:, :, j * 3, :], p=2)
                    # by scaling the distance by the inverse of the nieghborhood distance, we make sure everything is equidistant
                    ij_dist = ij_dist * (1 / abs(i - j))
                    inner_dists.append(ij_dist)
            mean_inner_dist = torch.mean(torch.stack(inner_dists, dim=1), dim=1)
            # inner penalty is the deviation from the mean squared
            inner_penalty = torch.mean(torch.square(torch.stack(inner_dists).squeeze() - mean_inner_dist.squeeze()))
            inner_penalties.append(inner_penalty)
        return torch.mean(torch.stack(inner_penalties))

    def loss_function(self,
                      reconstructions: Tensor,
                      gt_images: Tensor,
                      vq_loss: Tensor,
                      points: Tensor,
                      log_loss: bool = False,
                      raw_images: Tensor = None,
                      composite: list = None,
                      gt_outline: Tensor = None,
                      pred_outline: Tensor = None,
                      gt_bnw: Tensor = None,
                      pred_bnw: Tensor = None,
                      color_weights: Tensor = None,
                      recon_weight: float = 1.0,  # weight for the reconstruction loss
                      recon_bw_weight: float = 1.0,  # weight for the bw reconstruction loss
                      outline_weight: float = 1.0,  # weight for the bw reconstruction loss
                      **kwargs) -> dict:

        loss_dict = {}
        final_loss = 0

        if self.vq_method != "fsq":  # FSQ does not train the codebook
            loss_dict['VQ_Loss'] = vq_loss
            final_loss += vq_loss  # UPDATE LOSS HERE

        # white_filter_mask = (gt_images == 1.0000).all(dim=1).unsqueeze(1).expand(-1, 2, -1, -1)
        white_filter_mask = ~gt_bnw[:, 0].bool().unsqueeze(1).expand(-1, 2, -1, -1)

        # convert to LAB and normalise
        recon_lab = rgb_to_lab(reconstructions)
        gt_lab = rgb_to_lab(gt_images)
        recon_lab[:, 0] = recon_lab[:, 0] / 100
        recon_lab[:, 1:3] = (recon_lab[:, 1:3] + 128) / 255
        gt_lab[:, 0] = gt_lab[:, 0] / 100
        gt_lab[:, 1:3] = (gt_lab[:, 1:3] + 128) / 255

        # PYRAMID does MSE anyway, so either one or the other
        if "mse" in self.image_loss:
            raise NotImplementedError("MSE loss disabled because already included in the pyramid.")
            # recons_loss = F.mse_loss(reconstructions, gt_images)
        elif "pyramid" in self.image_loss:
            ab_loss, l_loss, pyramic_contributions = self.gaussian_pyramid_loss(
                recon_lab, gt_lab,
                down_sample_steps=3, log_loss=log_loss,
                pyramid_weights=kwargs["pyramid_weights"],
                color_weights=color_weights,
                color_masks=white_filter_mask,
            )
            loss_dict.update(pyramic_contributions)
        else:
            raise Exception("One of the following image loss functions must be used: ['mse', 'pyramid']")
        loss_dict['AB_Loss'] = ab_loss
        loss_dict['L_Loss'] = l_loss
        final_loss += (ab_loss * recon_weight) + (l_loss * recon_bw_weight)  # UPDATE LOSS HERE

        if "outline" in self.image_loss:
            outline_loss = F.l1_loss(
                rgb_to_grayscale(rgba_to_rgb(pred_outline)).squeeze(),
                gt_outline
            ) * 10
            loss_dict['Outline_loss'] = outline_loss
            final_loss += (outline_loss * outline_weight)

        if "composite" in self.image_loss and composite is not None:
            composite = torch.stack(composite, dim=0)
            composite_loss = F.mse_loss(raw_images, composite)
            final_loss += composite_loss  # UPDATE LOSS HERE
            loss_dict['Composite_Loss'] = composite_loss

        if self.geometric_constraint == "inner_distance":
            # if False:
            #     max_dist = torch.cdist(torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 1.0]]), p=2).item()
            #     mean_inner_distance_batched = self._get_mean_inner_distance(points)
            #     # loss is weighted by the mean of black pixels, so that short strokes are not penalized as much
            #     # ä FIXME removed the scaling FOR NOW
            #     mean_black_pixels_batched = (1 - gt_images).mean(dim=(1, 2, 3))
            #     geometric_loss = (1 - (mean_inner_distance_batched / max_dist))
            #     scaled_geometric_loss = (geometric_loss * mean_black_pixels_batched).mean()
            #     geometric_loss = geometric_loss.mean()
            # else:
            inner_distance_penalty = self._get_inner_distance_penalty(points)
            scaled_geometric_loss = inner_distance_penalty
            final_loss += self.alpha * scaled_geometric_loss  # UPDATE LOSS HERE
            loss_dict["geometric_Loss"] = inner_distance_penalty
            loss_dict[self.geometric_constraint + "_loss"] = self.alpha * scaled_geometric_loss

        loss_dict["loss"] = final_loss  # this is the one processed by the model
        return loss_dict

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    @torch.no_grad()
    def reconstruct(self,
                    patches: Tensor,
                    gt_center_positions: Tensor,
                    padded_individual_max_length: float,
                    local_stroke_width: float = None,  # FIXME right now this is actually global
                    rendered_w=128.,
                    return_shapes: bool = False,
                    return_local_points: bool = False) -> Union[Drawing, Tensor]:
        """
        Reconstructs the input patches and uses gt positions to assemble them into a full SVG. Can be used to observe quality degradation of the quantization process.

        Args:
            - patches (Tensor): Input patches to be reconstructed
            - gt_center_positions (Tensor): Ground truth center positions of the patches
            - padded_individual_max_length (float): Padded individual max length of the patches, usually is individual_max_length+2
            - local_stroke_width (float): effects only reconstructed SVG, override the prediction of the model with a fixed stroke width

        Returns:
            - reconstructed_drawing (Drawing): Reconstructed drawing (use to save svg)
            - rasterized_reconstructions (Tensor): Rasterized reconstructions
        """
        [reconstructions, input, all_points, vq_loss, visual_attribute_dict], logging_dict = self.forward(patches,
                                                                                                          logging=False,
                                                                                                          return_visual_attributes=True)
        # these need to be scaled with 72 to keep the original viewbox aspect ratios intact
        if gt_center_positions.max() < 1.0:
            gt_center_positions = gt_center_positions * 72

        global_shapes = calculate_global_positions(all_points, padded_individual_max_length, gt_center_positions)[:, 0]

        # scale back into [0,1] range
        if global_shapes.max() > 1.0:
            global_shapes = global_shapes / 72

        if local_stroke_width is not None:
            local_stroke_widths = torch.ones_like(visual_attribute_dict["stroke_widths"]) * local_stroke_width
        else:
            local_stroke_widths = width_pred_to_local_stroke_width(visual_attribute_dict["stroke_widths"],
                                                                   self.patch_size,
                                                                   padded_individual_max_length)
        global_stroke_widths = local_stroke_widths / padded_individual_max_length * 72
        visual_attribute_dict["local_stroke_widths"] = local_stroke_widths
        visual_attribute_dict["global_stroke_widths"] = global_stroke_widths

        try:
            # reconstructed_drawing = shapes_to_drawing(global_shapes, stroke_width=stroke_widths, w=rendered_w)
            # the misconception here is that we need global stroke width, which is WRONG. That would look as thick as the local strokes, which is not desired
            reconstructed_drawing = shapes_to_drawing(global_shapes, stroke_width=local_stroke_widths,
                                                      visual_attribute_dict=visual_attribute_dict, w=rendered_w)
            # rasterized_reconstructions = svg_string_to_tensor(reconstructed_drawing.tostring())
        except Exception as e:
            print("Error during reconstruction: ", e)
            print(f"Got max of: {global_shapes.max()} Limited shapes to [0,1]")
            global_shapes = torch.clamp(global_shapes, 0, 1)
            reconstructed_drawing = shapes_to_drawing(global_shapes, global_stroke_widths,
                                                      visual_attribute_dict=visual_attribute_dict, w=rendered_w)
            # rasterized_reconstructions = None
        if return_shapes:
            if return_local_points:
                return reconstructed_drawing, reconstructions, global_shapes, visual_attribute_dict, all_points
            return reconstructed_drawing, reconstructions, global_shapes, visual_attribute_dict
        else:
            return reconstructed_drawing, reconstructions