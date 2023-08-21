import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from x_transformers import Decoder
from thesis.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from thesis.models.simple_vector_decoder import SimpleVectorDecoder
from thesis.models.mlp import MultiLayerPerceptron

class VectorGPT(nn.Module):
    def __init__(self,
                    # vector_gpt_config: VectorGPTArgs,
                    image_encoder_model: str = "resnet18",
                    latent_transformer_dim: int = 512,
                    latent_transformer_depth: int = 8,
                    latent_transformer_heads: int = 8,
                    latent_transformer_layer_dropout: float = 0.1,
                    vector_decoder_latent_dim: int = 512,
                    vector_decoder_paths: int = 5,
                    vector_decoder_radius: int = 3,
                    vector_decoder_render_size: int = 128,
                    stop_predictor_dims: list = [768, 512],
                    stop_predictor_activation: str = "relu",
                    stop_predictor_num_classes: int = 1,
                    context_length: int = 25,
                    reconstruction_loss_weight: float = 0.7,
                    **kwargs
                 ):
        super(VectorGPT, self).__init__()

        self.image_encoder_model = image_encoder_model
        self.latent_transformer_dim = latent_transformer_dim
        self.latent_transformer_depth = latent_transformer_depth
        self.latent_transformer_heads = latent_transformer_heads
        self.latent_transformer_layer_dropout = latent_transformer_layer_dropout
        self.vector_decoder_latent_dim = vector_decoder_latent_dim
        self.vector_decoder_paths = vector_decoder_paths
        self.vector_decoder_radius = vector_decoder_radius
        self.vector_decoder_render_size = vector_decoder_render_size
        self.stop_predictor_dims = stop_predictor_dims
        self.stop_predictor_activation = stop_predictor_activation
        self.stop_predictor_num_classes = stop_predictor_num_classes
        self.context_length = context_length
        self.reconstruction_loss_weight = reconstruction_loss_weight

        if self.image_encoder_model == "resnet18":
            self.resnet = ResNet18(self.latent_transformer_dim)
        elif self.image_encoder_model == "resnet34":
            self.resnet = ResNet34(self.latent_transformer_dim)
        elif self.image_encoder_model == "resnet50":
            self.resnet = ResNet50(self.latent_transformer_dim)
        elif self.image_encoder_model == "resnet101":
            self.resnet = ResNet101(self.latent_transformer_dim)
        elif self.image_encoder_model == "resnet152":
            self.resnet = ResNet152(self.latent_transformer_dim)
        else:
            raise ValueError(f"[ERROR] You did not specify a correct Image Encoder. Expected something like 'resnet18', got {self.image_encoder_model}.")
        
        self.positional_embedding = nn.Embedding(self.context_length, self.latent_transformer_dim)
        self.latent_transformer = nn.Sequential(Decoder(dim=self.latent_transformer_dim,
                                                        depth=self.latent_transformer_depth,
                                                        heads=self.latent_transformer_heads,
                                                        layer_dropout=self.latent_transformer_layer_dropout), 
                                                nn.LayerNorm(self.latent_transformer_dim),
                                                nn.Linear(self.latent_transformer_dim, self.latent_transformer_dim))
        self.vector_decoder = SimpleVectorDecoder(latent_dim=self.vector_decoder_latent_dim,
                                                  paths=self.vector_decoder_paths,
                                                  radius=self.vector_decoder_radius,
                                                  render_size=self.vector_decoder_render_size)
        self.stop_predictor = MultiLayerPerceptron(input_dim=self.latent_transformer_dim,
                                                   dims=self.stop_predictor_dims,
                                                   activation=self.stop_predictor_activation,
                                                   num_classes=self.stop_predictor_num_classes)

    def forward(self, input_images: Tensor, drop_alpha_channel = False, **kwargs):
        """
        Expects images to be in (batch, timesteps, channel, width, height).
        Important: expects separate shape layers as input, not composite images.

        Outputs rasterized shape layers (b, t, c, w, h) and stop signals (b, t)
        """
        bs = input_images.size(0)
        timesteps = input_images.size(1)

        # first we encode. (b, t, c, w, h) -> (b, t, z)
        intermediate = [self.resnet(input_images[:,t,:,:]) for t in range(timesteps)]
        encoded_images = torch.stack(intermediate, dim=1) # (b, t, z)

        pos_embeddings = self.positional_embedding(torch.arange(timesteps).to(encoded_images.device)) # (t, z)
        encoded_images = encoded_images + pos_embeddings # (b, t, z)

        # then we transform (b, t, z) -> (b, t, z')
        transformed_latents = self.latent_transformer(encoded_images)

        # then we decode each t iteratively, TODO find out if it can be batchified
        rasterized_shapes = []
        stop_preds = []
        for t in range(timesteps):
            rasterized_shape, _, _, _ = self.vector_decoder(transformed_latents[:,t,:])
            rasterized_shapes.append(rasterized_shape)

            stop_pred = self.stop_predictor.to(transformed_latents.device).forward(transformed_latents[:,t,:])
            stop_preds.append(stop_pred)

        # re-introduce the time dimension
        rasterized_shapes = torch.stack(rasterized_shapes, dim=1) # (b, t, c, w, h)
        if drop_alpha_channel:
            rasterized_shapes = rasterized_shapes[:, :, :3, :, :]
        stop_preds = torch.stack(stop_preds, dim=1) # (b, t, 1)
        stop_preds = stop_preds.squeeze(-1) # (b, t)

        return rasterized_shapes, stop_preds

    def loss_function(self, gt_shape_layers: Tensor, pred_images: Tensor, gt_stop_signals: Tensor, stop_signals:Tensor, **kwargs):
        """
        Inputs:
        gt_shape_layers & pred_images in format (b, t, c, w, h)
        gt_stop_signals in format (b, t)
        stop signals in format (b, t)

        Important: gt_shape_layers are the individually rendered shapes for loss calculation, not the complete composition
        
        Outputs three losses: [final_loss, recons_loss, stop_prediction_loss]
        Precise formula TBD. Currently averages over time and batch dimension.
        """
        assert gt_shape_layers.size(1) == pred_images.size(1) == gt_stop_signals.size(1) == stop_signals.size(1), "Received different amount of timesteps for stop signals or images."

        # drop alpha channel for MSE loss calculation
        if pred_images.size(2) == 4:
            pred_images = pred_images[:, :, :3, :, :]
        
        # # mask out the loss calculation for stop loss beyond the first stop signal
        mask = gt_stop_signals >= 0.
        selected_gt_stop_signals = torch.masked_select(gt_stop_signals, mask)
        selected_stop_signals = torch.masked_select(stop_signals, mask)

        # mask out loss calculation for shape predictions from the first stop signal on
        mask = gt_stop_signals == 0.
        expanded_mask = mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        selected_gt_shape_layers = torch.masked_select(gt_shape_layers, expanded_mask)
        selected_pred_images = torch.masked_select(pred_images, expanded_mask)

        stop_prediction_loss = F.binary_cross_entropy(selected_stop_signals, selected_gt_stop_signals)
        recons_loss = F.mse_loss(selected_pred_images, selected_gt_shape_layers)
        # stop_prediction_loss = F.binary_cross_entropy(stop_signals, gt_stop_signals)
        # recons_loss = F.mse_loss(pred_images, gt_shape_layers)

        final_loss = (1 - self.reconstruction_loss_weight)*stop_prediction_loss + self.reconstruction_loss_weight*recons_loss
        
        return final_loss, recons_loss, stop_prediction_loss