from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import wandb
from x_transformers import Decoder
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .simple_vector_decoder import SimpleVectorDecoder
from .mlp_vector_head import MLPVectorHeadFixed, MLPRasterHead
from .mlp import MultiLayerPerceptron
import kornia
from thesis.utils import log_all_images

class PositionalEncoding(nn.Module):
    """
    Non-learnable positional encoding for Transformer. Taken from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:
        - d_model (int): Dimensionality of the model
        - dropout (float): Dropout rate
        - max_len (int): Maximum length of the input sequence (context length)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class VectorGPT(nn.Module):
    """
    weights of image_encoder_ckpt_path must have keys in the form of "conv1.weight", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var"
    """
    def __init__(self,
                    image_encoder_model: str = "resnet18",
                    image_encoder_ckpt_path: str = None,
                    image_encoder_latent_dim: int = 128,
                    skip_transformer: str = False,
                    learnable_positional_encoding: bool = False,
                    latent_transformer_dim: int = 512,
                    latent_transformer_depth: int = 8,
                    latent_transformer_heads: int = 8,
                    latent_transformer_layer_dropout: float = 0.1,
                    vector_decoder_model: str = "cnn",
                    vector_decoder_latent_dim: int = 512,
                    vector_decoder_paths: int = 5,
                    vector_decoder_radius: int = 3,
                    vector_decoder_render_size: int = 128,
                    vector_decoder_filled: bool = True,
                    vector_decoder_max_stroke_width: float = 10.0,
                    stop_predictor_dims: list = [768, 512],
                    stop_predictor_activation: str = "relu",
                    stop_predictor_num_classes: int = 1,
                    context_length: int = 25,
                    reconstruction_loss_weight: float = 0.7,
                    loss_mode = None,
                    down_sample_steps: int = 3,
                    wandb_logging: bool = False,
                    **kwargs
                 ):
        super(VectorGPT, self).__init__()

        self.image_encoder_model = image_encoder_model
        self.image_encoder_ckpt_path = image_encoder_ckpt_path
        self.image_encoder_latent_dim = image_encoder_latent_dim
        self.skip_transformer = skip_transformer
        self.learnable_positional_encoding = learnable_positional_encoding
        self.latent_transformer_dim = latent_transformer_dim
        self.latent_transformer_depth = latent_transformer_depth
        self.latent_transformer_heads = latent_transformer_heads
        self.latent_transformer_layer_dropout = latent_transformer_layer_dropout
        self.vector_decoder_model = vector_decoder_model.lower()
        self.vector_decoder_latent_dim = vector_decoder_latent_dim
        self.vector_decoder_paths = vector_decoder_paths
        self.vector_decoder_radius = vector_decoder_radius
        self.vector_decoder_render_size = vector_decoder_render_size
        self.vector_decoder_filled = vector_decoder_filled
        self.vector_decoder_max_stroke_width = vector_decoder_max_stroke_width
        self.stop_predictor_dims = stop_predictor_dims
        self.stop_predictor_activation = stop_predictor_activation
        self.stop_predictor_num_classes = stop_predictor_num_classes
        self.context_length = context_length
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.loss_mode = loss_mode
        self.down_sample_steps = down_sample_steps
        self.wandb_logging = wandb_logging

        assert self.loss_mode in [None, "default", "pyramid", "merged", "pyramid+merged"], f"Loss mode {self.loss_mode} not supported."

        if self.image_encoder_model == "resnet18":
            self.resnet = ResNet18(self.image_encoder_latent_dim)
        elif self.image_encoder_model == "resnet34":
            self.resnet = ResNet34(self.image_encoder_latent_dim)
        elif self.image_encoder_model == "resnet50":
            self.resnet = ResNet50(self.image_encoder_latent_dim)
        elif self.image_encoder_model == "resnet101":
            self.resnet = ResNet101(self.image_encoder_latent_dim)
        elif self.image_encoder_model == "resnet152":
            self.resnet = ResNet152(self.image_encoder_latent_dim)
        else:
            raise ValueError(f"[ERROR] You did not specify a correct Image Encoder. Expected something like 'resnet18', got {self.image_encoder_model}.")

        if self.image_encoder_ckpt_path is not None:
            missing, unexpexted =  self.resnet.load_state_dict(torch.load(self.image_encoder_ckpt_path), strict=False)
            print(f"[INFO] Successfully loaded weights from {self.image_encoder_ckpt_path}")
            print(f"[INFO] {len(missing)} missing keys.")
            print(f"[INFO] {len(unexpexted)} unexpected keys.")

        if self.learnable_positional_encoding:
            self.positional_embedding = nn.Embedding(self.context_length, self.latent_transformer_dim)
        else:
            self.positional_embedding = PositionalEncoding(self.latent_transformer_dim,
                                                           dropout=self.latent_transformer_layer_dropout,
                                                           max_len=self.context_length)
        self.image_latent_to_transformer_latent = nn.Linear(self.image_encoder_latent_dim, self.latent_transformer_dim)
        self.latent_transformer = nn.Sequential(Decoder(dim=self.latent_transformer_dim,
                                                        depth=self.latent_transformer_depth,
                                                        heads=self.latent_transformer_heads,
                                                        layer_dropout=self.latent_transformer_layer_dropout), 
                                                nn.LayerNorm(self.latent_transformer_dim),
                                                nn.Linear(self.latent_transformer_dim, self.latent_transformer_dim))
        
        self.z_order = nn.Sequential(
            nn.Linear(self.latent_transformer_dim, self.latent_transformer_dim),
            nn.ReLU(),  # bound spatial extent
            nn.Linear(self.latent_transformer_dim, self.latent_transformer_dim),
            nn.ReLU(),  # bound spatial extent
            nn.Linear(self.latent_transformer_dim, 1),
        )
        self.transformer_latent_to_vector_decoder_input = nn.Linear(self.latent_transformer_dim, self.vector_decoder_latent_dim)

        if self.vector_decoder_model == "cnn":
            self.vector_decoder = SimpleVectorDecoder(latent_dim=self.vector_decoder_latent_dim,
                                                    paths=self.vector_decoder_paths,
                                                    radius=self.vector_decoder_radius,
                                                    render_size=self.vector_decoder_render_size,
                                                    filled=self.vector_decoder_filled)
        elif self.vector_decoder_model == "mlp":
            self.vector_decoder = MLPVectorHeadFixed(latent_dim=self.vector_decoder_latent_dim,
                                                segments=self.vector_decoder_paths,
                                                imsize=self.vector_decoder_render_size,
                                                max_stroke_width=self.vector_decoder_max_stroke_width)
        elif self.vector_decoder_model == "raster_mlp":
            self.vector_decoder = MLPRasterHead(latent_dim=self.vector_decoder_latent_dim,
                                                render_size=self.vector_decoder_render_size)
        else:
            raise ValueError("You did not specify a correct Vector Decoder. Expected something like 'cnn' or 'mlp'. Check your config.")
        self.stop_predictor = MultiLayerPerceptron(input_dim=self.latent_transformer_dim,
                                                dims=self.stop_predictor_dims,
                                                activation=self.stop_predictor_activation,
                                                num_classes=self.stop_predictor_num_classes)
        
    def soft_composite(self, layers: list, z_layers: list):
        """
        Differentiable compositing implementation by Im2Vec authors.
        See: https://arxiv.org/abs/2010.08788

        Args:
            - layers (list): rasterized shape layers
            - z_layers (list): z-order of the shape layers
        
        Returns:
            - rgb (Tensor): composite image
        """
        n = len(layers)

        inv_mask = (1 - layers[0][:, 3:4, :, :])
        for i in range(1, n):
            inv_mask = inv_mask * (1 - layers[i][:, 3:4, :, :])

        sum_alpha = layers[0][:, 3:4, :, :] * z_layers[0]
        for i in range(1, n):
            sum_alpha = sum_alpha + layers[i][:, 3:4, :, :] * z_layers[i]
        sum_alpha = sum_alpha + inv_mask

        inv_mask = inv_mask / sum_alpha

        rgb = layers[0][:, :3] * layers[0][:, 3:4, :, :] * z_layers[0] / sum_alpha
        for i in range(1, n):
            rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * z_layers[i] / sum_alpha
        rgb = rgb * (1 - inv_mask) + inv_mask
        return rgb

    def forward(self, input_images: Tensor, drop_alpha_channel = False, verbose = False,**kwargs):
        """
        Expects images to be in (batch, timesteps, channel, width, height).
        Important: expects separate shape layers as input, not composite images.

        Outputs 
            - rasterized shape layers (b, t, c, w, h) 
            - stop signals (b, t)
            - merged_preds (b, t, c, w, h) 
        """
        timesteps = input_images.size(1)
        merged_preds = None

        # first we encode. (b, t, c, w, h) -> (b, t, z)
        intermediate = [self.resnet(input_images[:,t,:,:,:]) for t in range(timesteps)]
        encoded_images = torch.stack(intermediate, dim=1) # (b, t, z)

        # map the latent dimensions
        encoded_images = self.image_latent_to_transformer_latent(encoded_images)
        
        if self.learnable_positional_encoding:
            pos_embeddings = self.positional_embedding(torch.arange(timesteps).to(encoded_images.device)) # (t, z)
            encoded_images = encoded_images + pos_embeddings # (b, t, z) broadcast should work here
        else:
            encoded_images = self.positional_embedding(encoded_images)  # addition happens in the module

        # then we transform (b, t, z) -> (b, t, z')
        if self.skip_transformer:
            transformed_latents = encoded_images
        else:
            transformed_latents = self.latent_transformer(encoded_images)

        # map the latent dimensions
        transformed_latents = self.transformer_latent_to_vector_decoder_input(transformed_latents)

        # then we decode each t iteratively
        rasterized_shapes = []
        stop_preds = []
        z_layers = []
        merged_preds = []
        for t in range(timesteps):
            out = self.vector_decoder.forward(transformed_latents[:,t,:], verbose=verbose)
            rasterized_shape = out[0]
            stop_pred = self.stop_predictor.to(transformed_latents.device).forward(transformed_latents[:,t,:])
            rasterized_shapes.append(rasterized_shape)
            stop_preds.append(stop_pred)
            if "merged" in self.loss_mode:
                z_pred = self.z_order.to(transformed_latents.device).forward(transformed_latents[:,t,:])
                z_layers.append(torch.exp(z_pred[:, :, None, None]))
                merged_preds.append(self.soft_composite(rasterized_shapes, z_layers))

        # re-introduce the time dimension
        rasterized_shapes = torch.stack(rasterized_shapes, dim=1) # (b, t, c, w, h)
        if len(merged_preds) > 0:
            merged_preds = torch.stack(merged_preds, dim=1) # (b, t, c, w, h), always 3 channels atm
        else:
            merged_preds = None
        if drop_alpha_channel:
            rasterized_shapes = rasterized_shapes[:, :, :3, :, :]

        stop_preds = torch.stack(stop_preds, dim=1) # (b, t, 1)
        stop_preds = stop_preds.squeeze(-1) # (b, t)

        return rasterized_shapes, stop_preds, merged_preds
    
    def transform_loss_tensor_to_image(self, loss_tensor: Tensor):
        """
        Transforms a loss tensor to an image tensor for logging purposes.

        Args:
            - loss_tensor (Tensor): Loss tensor in format (-1, c, w, h)

        Returns:
            - loss_image (Tensor): colored loss image in format (-1, c, w, h)
        """
        cm = plt.get_cmap("Reds")
        loss_tensor = loss_tensor.mean(dim=1)  # mean the channel dimension
        loss_image = torch.from_numpy(cm(loss_tensor.detach().cpu())).permute(0,3,1,2)
        return loss_image

    def gaussian_pyramid_loss(self, recons_images: Tensor, gt_images: Tensor, down_sample_steps: int = 3, log_loss: bool = False):
        """
        Calculates the gaussian pyramid loss between reconstructed images and ground truth images.

        Args:
            - recons_images (Tensor): Reconstructed images in format (-1, c, w, h)
            - gt_images (Tensor): Ground truth images in format (-1, c, w, h)
            - down_sample_steps (int): Number of downsample steps to calculate the loss for. Default: 3

        Returns:
            - recon_loss (Tensor): The gaussian pyramid loss between reconstructed images and ground truth images.
        """
        dsample = kornia.geometry.transform.pyramid.PyrDown()
        timesteps_to_log = 4
        recon_loss = F.mse_loss(recons_images, gt_images, reduction='none')
        recons_loss_contributions = {}
        if log_loss:
            all_loss_images = []
            all_loss_images.append(self.transform_loss_tensor_to_image(recon_loss[:timesteps_to_log]))
        recon_loss = recon_loss.mean()
        for j in range(2, 2 + down_sample_steps):
            weight = 1 / j
            recons_images = dsample(recons_images)
            gt_images = dsample(gt_images)
            loss_images = F.mse_loss(recons_images, gt_images, reduction='none')
            if log_loss:
                all_loss_images.append(self.transform_loss_tensor_to_image(loss_images[:timesteps_to_log]))

            curr_pyramid_loss = loss_images.mean() / weight
            recons_loss_contributions[f"pyramid_loss_step_{j-1}"] = curr_pyramid_loss
            recon_loss = recon_loss + curr_pyramid_loss

        if log_loss:
            log_all_images(all_loss_images, log_key="pyramid loss", caption=f"Gaussian Pyramid Loss, {down_sample_steps+1} steps")
            wandb.log(recons_loss_contributions)
        return recon_loss

    def loss_function(self, gt_shape_layers: Tensor, pred_images: Tensor, gt_stop_signals: Tensor, gt_merged_targets: Tensor, merged_preds:Tensor, stop_signals:Tensor, log_loss: bool = False, **kwargs):
        """
        Args:
            - gt_shape_layers & pred_images in format (b, t, c, w, h)
            - gt_merged_targets & merged_preds in format (b, t, c, w, h)
            - gt_stop_signals in format (b, t)
            - stop signals in format (b, t)
            - log_loss (bool): Whether to log the loss images to wandb. Default: False

        Important: gt_shape_layers are the individually rendered shapes for loss calculation. The complete compositions up to each current timestep is captured in gt_merged_targets.
        
        Outputs three losses: [final_loss, recons_loss, stop_prediction_loss]
        Precise formula TBD. Currently averages over time and batch dimension.
        """
        assert gt_shape_layers.size(1) == pred_images.size(1) == gt_stop_signals.size(1) == stop_signals.size(1), "Received different amount of timesteps for stop signals or images."

        # drop alpha channel for MSE loss calculation
        if pred_images.shape[2] == 4:
            pred_images = pred_images[:, :, :3, :, :]

        bs, t, c, w, h = pred_images.shape

        # mask out the loss calculation for stop loss beyond the first stop signal
        mask = gt_stop_signals >= 0.
        selected_gt_stop_signals = torch.masked_select(gt_stop_signals, mask)
        selected_stop_signals = torch.masked_select(stop_signals, mask)

        # mask out loss calculation for shape predictions from the first stop signal on
        mask = gt_stop_signals == 0.
        mask = mask.view(*mask.shape, 1, 1, 1)  # ensure broadcasting
        selected_gt_shape_layers = torch.masked_select(gt_shape_layers, mask).view(-1, c, w, h)
        selected_pred_images = torch.masked_select(pred_images, mask).view(-1, c, w, h)

        if merged_preds is not None and gt_merged_targets is not None:
            selected_gt_merged_targets = torch.masked_select(gt_merged_targets, mask).view(-1, c, w, h)
            selected_merged_preds = torch.masked_select(merged_preds, mask).view(-1, c, w, h)

        if self.loss_mode == "pyramid":
            recons_loss = self.gaussian_pyramid_loss(selected_pred_images, selected_gt_shape_layers, log_loss = log_loss, down_sample_steps=self.down_sample_steps)  # logging happens in this function automatically
        elif self.loss_mode == "merged":
            recons_loss = self.gaussian_pyramid_loss(selected_merged_preds, selected_gt_merged_targets, log_loss = log_loss, down_sample_steps=self.down_sample_steps)
        elif self.loss_mode == "pyramid+merged":
            recons_loss_shapes = self.gaussian_pyramid_loss(selected_pred_images, selected_gt_shape_layers, log_loss = log_loss, down_sample_steps=self.down_sample_steps)  # logging happens in this function automatically
            recons_loss_merged = self.gaussian_pyramid_loss(selected_merged_preds, selected_gt_merged_targets, log_loss = log_loss, down_sample_steps=self.down_sample_steps)
            recons_loss = (recons_loss_shapes + recons_loss_merged) / 2
        else:
            recons_loss = F.mse_loss(selected_pred_images, selected_gt_shape_layers, reduction="none")  # no reduction to log loss images
            if log_loss:
                log_all_images([self.transform_loss_tensor_to_image(recons_loss)], log_key="reconstruction loss", caption="Reconstruction Loss MSE")

        stop_prediction_loss = F.binary_cross_entropy(selected_stop_signals, selected_gt_stop_signals)
        recons_loss = recons_loss.mean()

        final_loss = (1 - self.reconstruction_loss_weight)*stop_prediction_loss + self.reconstruction_loss_weight*recons_loss

        return final_loss, recons_loss, stop_prediction_loss
