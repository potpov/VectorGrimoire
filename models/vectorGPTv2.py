from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import wandb
from x_transformers import Decoder
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.simple_vector_decoder import SimpleVectorDecoder
from models.mlp_vector_head import MLPVectorHeadFixed, MLPRasterHead
from models.mlp import MultiLayerPerceptron
import kornia
from utils import log_all_images
import pydiffvg


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
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

class VectorGPTv2(nn.Module):
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
                    vector_decoder_primitive: str = "cubic",
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
        super(VectorGPTv2, self).__init__()

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
        self.vector_decoder_primitive = vector_decoder_primitive.lower()
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

        assert self.loss_mode in [None, "default", "pyramid"], f"Loss mode {self.loss_mode} not supported. Merged was deprecated in v2."
        assert self.vector_decoder_primitive in ["cubic", "linear"], f"Vector decoder primitive {self.vector_decoder_primitive} not supported. Expected 'cubic', 'quadratic' or 'linear'."

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
        self.shape_position_to_vector_decoder_input = nn.Linear(4, self.vector_decoder_latent_dim)

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
        self.position_predictor = MultiLayerPerceptron(input_dim=self.latent_transformer_dim,
                                                dims=[768, 512, 256],
                                                activation="relu",
                                                num_classes=2)
        
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

    def forward(self, input_absolute_images: Tensor, input_positions: Tensor, only_final_timestep: bool = False, drop_alpha_channel = False, verbose = False,**kwargs):
        """
        Expects images to be in (batch, timesteps, channel, width, height).
        input_positions are in shape (batch, timesteps, 4): where the 4 is split into: rel_start_point_x, rel_start_point_y, rel_end_point_x, rel_end_point_y

        Outputs 
            - rasterized shape layers (b, t, c, w, h) 
            - stop signals (b, t)
            - merged_preds (b, t, c, w, h) 
        """
        timesteps = input_absolute_images.size(1)
        merged_preds = None

        # first we encode. (b, t, c, w, h) -> (b, t, z)
        intermediate = [self.resnet(input_absolute_images[:,t,:,:,:]) for t in range(timesteps)]
        encoded_images = torch.stack(intermediate, dim=1) # (b, t, z)

        # map the latent dimensions
        encoded_images = self.image_latent_to_transformer_latent(encoded_images)
        embedded_shape_positions = self.shape_position_to_vector_decoder_input(input_positions)
        
        if self.learnable_positional_encoding:
            pos_embeddings = self.positional_embedding(torch.arange(timesteps).to(encoded_images.device)) # (t, z)
            encoded_images = encoded_images + pos_embeddings # (b, t, z) broadcast should work here
        else:
            encoded_images = self.positional_embedding(encoded_images)  # addition happens in the module

        # TODO add an extension of the dimension instead of addition here, modes "add" and "concat"
        # add the embedding of the positions of the previous shape
        encoded_images = encoded_images + embedded_shape_positions

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
        pos_preds = []
        svg_points = []
        z_layers = []
        merged_preds = []
        stroke_widths = []
        for t in range(timesteps):
            if only_final_timestep and t != timesteps - 1:
                continue
            if self.vector_decoder_model == "mlp":
                out = self.vector_decoder.forward(transformed_latents[:,t,:], primitive = self.vector_decoder_primitive, verbose=verbose)
            else:
                out = self.vector_decoder.forward(transformed_latents[:,t,:], verbose=verbose)
            rasterized_shape = out[0]
            stroke_width = out[-1]
            bezier_points = out[-2]  # (b, 1, self.segments*3+1, 2)
            svg_points.append(bezier_points)
            stroke_widths.append(stroke_width)
            stop_pred = self.stop_predictor.to(transformed_latents.device).forward(transformed_latents[:,t,:])
            pos_pred = self.position_predictor.to(transformed_latents.device).forward(transformed_latents[:,t,:])
            rasterized_shapes.append(rasterized_shape)
            stop_preds.append(stop_pred)
            pos_preds.append(pos_pred)
            if "merged" in self.loss_mode:
                z_pred = self.z_order.to(transformed_latents.device).forward(transformed_latents[:,t,:])
                z_layers.append(torch.exp(z_pred[:, :, None, None]))
                merged_preds.append(self.soft_composite(rasterized_shapes, z_layers))

        if only_final_timestep:
            rasterized_shape = rasterized_shapes[-1]  # (b, c, w, h)
            stop_pred = stop_preds[-1]  # (b, t)
            pos_pred = pos_preds[-1]  # 
            bezier_pred = svg_points[-1]
            stroke_width = stroke_widths[-1]
            if len(merged_preds) > 0:
                merged_pred = merged_preds[-1]
            else:
                merged_pred = None
            if drop_alpha_channel:
                rasterized_shape = rasterized_shape[:, :3, :, :]
            return rasterized_shape, stop_pred, merged_pred, bezier_pred, stroke_width, pos_pred

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

        pos_preds = torch.stack(pos_preds, dim=1)  # (b, t, 2)

        return rasterized_shapes, stop_preds, merged_preds, pos_preds
    
    @torch.no_grad()
    def generate(self, images: Tensor, positions: Tensor, max_new_steps: int):
        """
        Note: currently this takes images as input, but to fully assemble, you would ofc need SVG shapes as input.
        Currently does not move the positional embedding if context length is exceeded. So it will always assume t=0 for the first input image.

        images: (b, t_middle, c, w, h)
        positions: (b, t_end, 4)

        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        raise NotImplementedError("This is not implemented yet. Please use the generate from svg function.")
        assert images.size(0) == 1, "Currently only batch size 1 is supported for generation."

        start_time = images.size(1)

        for t in range(max_new_steps):
            print(f"Generating step {start_time+t+1} of {start_time+max_new_steps}")
            # if the sequence context is growing too long we must crop it at block_size
            images_cond = images if images.size(1) <= self.context_length else images[:, -self.context_length:]

            curr_positions = positions[:, :images.size(1)]
            curr_positions = curr_positions if curr_positions.size(1) <= self.context_length else curr_positions[:, -self.context_length:]
            # forward the model to get the logits for the index in the sequence
            rasterized_shape, stop_pred, _, bezier_pred, pos_pred = self.forward(images_cond, curr_positions, only_final_timestep=True, drop_alpha_channel=True)
            rasterized_shape = rasterized_shape[:, None, :, :, :]  # (b, 1, c, w, h)

            # (optional) sample here

            # append sampled image to the running sequence and continue
            print(images.shape, rasterized_shape.shape)
            images = torch.cat((images, rasterized_shape), dim=1)
            # positions = torch.cat((positions, rasterized_shape), dim=1)

            # stop if stop_pred is above threshold
            print("stop_pred: ", stop_pred[0, 0])
            if stop_pred[0, 0] > 0.70:
                print("REACHED STOP SIGNAL")
                return images

        return images
    
    @torch.no_grad()
    def generate_from_svg(self, input_bezier_points: Tensor, input_bezier_widths: Tensor, max_new_steps: int, scale: float, positions: Tensor, mode="auto_regressive"):
        """
        We assume input mode is absolute_merged here. Does perform padding of the input to match the training input.

        Input:
            - input_bezier_points in format (b, t, 4, 2) - each timestep are the points of one cubic bezier curve, relative positions between [0, 1]
            - input_bezier_widths in format (b, t, 1) - stroke widths for each timestep
            - max_new_steps: number of steps to generate
            - scale - the ratio of the full svg bounding box and the individual centered svg bounding box
            - positions - the start and end points for start/end points of beziers in format (b, t, 4)
            - mode - how to deal with predictions, either append them and use as input ("auto-regressive") or just save them but use GT as input ("teacher-forcing")
        """
        assert mode in ["auto_regressive", "teacher_forcing", "no_input"], f"Mode {mode} not supported. Expected 'auto-regressive' or 'teacher-forcing' or 'no_input'."
        assert scale > 1.0, "Scale must be larger than 1.0, did you calculate it the wrong way around?"
        assert input_bezier_points.size(0) == 1, "Currently only batch size 1 is supported for generation."
        assert input_bezier_points.size(2) == 4, "Expected input_bezier_points to be in format (b, t, 4, 2)"
        assert input_bezier_points.dim() == 4, "Expected input_bezier_points to be in format (b, t, 4, 2)"
        assert positions.size(1) == input_bezier_points.size(1), f"Expected positions ({positions.size(1)}) and input_bezier_points ({input_bezier_points.size(1)}) to have the same number of timesteps."
        
        return_tuple = None

        absolute_merged_inputs = []
        for t in range(1, input_bezier_points.size(1)):
            # TODO change this behaviour from [:t] to [t] for individual shape layers instead of merged ones
            curr_merged_absolute_input, _ = self._bezier_render(input_bezier_points[:, :t, :, :], 
                                                        input_bezier_widths[:, :t, :], 
                                                        torch.ones(*input_bezier_widths[:, :t, :].shape), 
                                                        canvas_size=128, 
                                                        primitive = "cubic", 
                                                        colors=None, 
                                                        white_background=True)
            
            absolute_merged_inputs.append(curr_merged_absolute_input[:,:3,:,:])
        
        # this is the Ground Truth throughout the generation process
        absolute_merged_inputs = torch.stack(absolute_merged_inputs, dim=1)  # (b, t, c, w, h)
        
        # add start padding that was also used in training
        pad_input = torch.ones(*absolute_merged_inputs[:,0].shape).unsqueeze(1)
        pad_pos = torch.zeros(*positions[:,0].shape).unsqueeze(1)

        positions = torch.cat((pad_pos, positions), dim=1)
        absolute_merged_inputs = torch.cat((pad_input, absolute_merged_inputs), dim=1)

        all_gt_bezier_points = input_bezier_points  # (b, t, 4, 2)
        all_gt_widths = input_bezier_widths  # (b, t, 1)

        # always start at the middle of all timesteps
        start_time = absolute_merged_inputs.size(1) // 2

        # this tracks all the generations
        generations = absolute_merged_inputs[:, :start_time, :, :, :].clone()
        # teacher_forcing_generations = absolute_merged_inputs[:, :start_time, :, :, :].clone()
        bezier_predictions = input_bezier_points[:, :start_time, :, :].clone()
        width_predictions = input_bezier_widths[:, :start_time, :].clone()

        all_rasterized_shapes = []
        for t in range(max_new_steps):
            print(f"Generating step {start_time+t+1}/{start_time*2} with max of {start_time+max_new_steps}")
            # if the sequence context is growing too long we must crop it at block_size
            if mode == "auto_regressive":
                curr_auto_regressive_generations = generations
                curr_input = curr_auto_regressive_generations if curr_auto_regressive_generations.size(1) <= self.context_length else curr_auto_regressive_generations[:, -self.context_length:]
            elif mode == "teacher_forcing":
                curr_gt_generations = absolute_merged_inputs[:, :start_time+t, :, :, :]
                curr_input = curr_gt_generations if curr_gt_generations.size(1) <= self.context_length else curr_gt_generations[:, -self.context_length:]
            elif mode == "no_input":
                curr_input = torch.ones(*generations.shape, device=generations.device)
                curr_input = curr_input if curr_input.size(1) <= self.context_length else curr_input[:, -self.context_length:]

            curr_positions = positions[:, :curr_input.size(1)]
            curr_positions = curr_positions if curr_positions.size(1) <= self.context_length else curr_positions[:, -self.context_length:]
            # forward the model to get the logits for the index in the sequence
            rasterized_shape, stop_pred, _, bezier_pred, stroke_width, pos_pred = self.forward(curr_input, curr_positions, only_final_timestep=True, drop_alpha_channel=True)
            all_rasterized_shapes.append(rasterized_shape)
            # bezier_pred# (b, 1, self.segments*3+1, 2) 

            # print("predicted stroke width", stroke_width)
            stroke_width = torch.tensor([[2.0]])

            # scale and shift bezier_pred to fit the absoulte_merged coordinates
            bezier_pred = bezier_pred / scale
            # stroke_width = stroke_width / scale

            # TODO this code was used for correct positioning. But it is not needed anymore since we now use predicted coordinates
            # start_offset_x = positions[:, start_time+t, 0] - bezier_pred[:,0,0,0]
            # start_offset_y = positions[:, start_time+t, 1] - bezier_pred[:,0,0,1]

            # use predicted coordinates
            start_offset_x = pos_pred[:, start_time+t, 0] - bezier_pred[:,0,0,0]
            start_offset_y = pos_pred[:, start_time+t, 1] - bezier_pred[:,0,0,1]

            bezier_pred[:, :,:, 0] = bezier_pred[:, :,:, 0] + start_offset_x[:, None]
            bezier_pred[:, :,:, 1] = bezier_pred[:, :,:, 1] + start_offset_y[:, None]

            # append to all_bezier_points
            bezier_predictions = torch.cat([bezier_predictions, bezier_pred], dim=1)
            width_predictions = torch.cat((width_predictions, stroke_width[:, None, :]), dim=1)
            # if mode == "auto_regressive" or mode == "no_input":
            #     bezier_predictions = torch.cat([bezier_predictions, bezier_pred], dim=1)
            #     width_predictions = torch.cat((width_predictions, stroke_width[:, None, :]), dim=1)
            # elif mode == "teacher_forcing":
            #     bezier_predictions = torch.cat([all_gt_bezier_points[:, :start_time+t, :], bezier_pred], dim=1)
            #     width_predictions = torch.cat((all_gt_widths[:, :start_time+t, :], stroke_width[:, None, :]), dim=1)
            
            curr_positioned_pred_shape_rasterized, _ = self._bezier_render(bezier_predictions, 
                                                    width_predictions, 
                                                    torch.ones(*width_predictions.shape), 
                                                    canvas_size=128, 
                                                    primitive = "cubic", 
                                                    colors=None, 
                                                    white_background=True)

            # (optional) sample here

            # append sampled image to the running sequence and continue
            curr_positioned_pred_shape_rasterized = curr_positioned_pred_shape_rasterized[:,:3,:,:].unsqueeze(0)
            generations = torch.cat((generations, curr_positioned_pred_shape_rasterized), dim=1)
            # absolute_merged_inputs = torch.cat((absolute_merged_inputs, curr_merged_absolute_input[:, None, :3, :, :]), dim=1)

            # positions = torch.cat((positions, rasterized_shape), dim=1)

            return_tuple = (generations, torch.stack(all_rasterized_shapes, dim=1), bezier_predictions, width_predictions, curr_merged_absolute_input[:,:3,:,:])

            # stop if stop_pred is above threshold
            print("stop_pred: ", stop_pred[0, 0])
            if stop_pred[0, 0] > 0.50:
                print("REACHED STOP SIGNAL")
                return return_tuple
            if start_time + t >= positions.size(1) - 1:
                print("start_time + t: ", start_time + t)
                print("positions.size(1): ", positions.size(1))
                print("REACHED MAX TIMESTEPS")
                return return_tuple
        return return_tuple
        
    def _render(self,
                canvas_width, 
                canvas_height, 
                shapes, 
                shape_groups, 
                samples=2,
                seed=42):
        
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, canvas_height, samples, samples,
                    seed,   # seed
                    None,  # background image
                    *scene_args)
        return img
    
    def _bezier_render(self, all_points: Tensor, all_widths: Tensor, all_alphas: Tensor,
                    canvas_size=32, primitive: str = "cubic", colors=None, white_background=True):
        device = all_points.device

        # all_points = 0.5*(all_points + 1.0) * canvas_size
        all_points = all_points * canvas_size

        eps = 1e-4
        all_points = all_points + eps*torch.randn_like(all_points, device=device)

        bs, num_strokes, num_pts, _ = all_points.shape
        num_segments = (num_pts - 1) // 3
        n_out = 4
        output = torch.zeros(bs, n_out, canvas_size, canvas_size,
                        device=device)

        scenes = []
        for batch in range(bs):
            shapes = []
            shape_groups = []
            for p in range(num_strokes):
                points = all_points[batch, p].contiguous()  # (num_pts, 2)
                if primitive == "cubic":
                    num_ctrl_pts = torch.zeros(num_segments, dtype=torch.int32) + 2
                elif primitive == "linear":
                    if num_segments > 1:
                        raise NotImplementedError("Linear primitive only supports 1 segment atm")
                    num_ctrl_pts = torch.zeros(num_segments, dtype=torch.int32)
                    points = points[[0, 3]]
                else:
                    raise NotImplementedError(f"Primitive {primitive} not implemented")
                width = all_widths[batch, p]
                alpha = all_alphas[batch, p]
                if colors is not None:
                    color = colors[batch, p]
                else:
                    color = torch.zeros(3, device=device)

                color = torch.cat([color, alpha.view(1,)])

                path = pydiffvg.Path(
                    num_control_points=num_ctrl_pts, points=points,
                    stroke_width=width, is_closed=False)
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=None,
                    stroke_color=color)
                shape_groups.append(path_group)

            # Rasterize
            scenes.append((canvas_size, canvas_size, shapes, shape_groups))
            raster = self._render(canvas_size, canvas_size, shapes, shape_groups,
                            samples=2)
            raster = raster.permute(2, 0, 1).view(4, canvas_size, canvas_size)

            # alpha = raster[3:4]
            # if colors is not None:  # color output
            #     image = raster[:3]
            #     alpha = alpha.repeat(3, 1, 1)
            # else:
            #     image = raster[:1]

            # # alpha compositing
            # image = image*alpha
            # output[k] = image
            output[batch] = raster

        output = output.to(device)
        
        if white_background:
            alpha = output[:, 3:4, :, :]
            output_white_bg = output[:, :3, :, :] * alpha + (1 - alpha)
            output = torch.cat([output_white_bg, alpha], dim=1)

        return output, scenes
    
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

    def loss_function(self,
                    gt_shape_layers: Tensor,
                    pred_images: Tensor,
                    gt_stop_signals: Tensor,
                    gt_merged_targets: Tensor,
                    merged_preds:Tensor,
                    stop_signals:Tensor,
                    position_predictions: Tensor,
                    gt_positions: Tensor,
                    log_loss: bool = False,
                    **kwargs):
        """
        Args:
            - gt_shape_layers & pred_images in format (b, t, c, w, h)
            - gt_merged_targets & merged_preds in format (b, t, c, w, h)
            - gt_stop_signals in format (b, t)
            - stop signals in format (b, t)
            - position_predictions and gt_positions in format (b, t, 2) and denote the position of the first point of the primitive
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
            # log the loss images here explicitly because it normally happens in the gaussian_pyramid_loss function
            if log_loss:
                log_all_images([self.transform_loss_tensor_to_image(recons_loss)], log_key="reconstruction loss", caption="Reconstruction Loss MSE")

        stop_prediction_loss = F.binary_cross_entropy(selected_stop_signals, selected_gt_stop_signals)
        recons_loss = recons_loss.mean()
        position_loss = F.mse_loss(position_predictions, gt_positions)

        final_loss = (1 - self.reconstruction_loss_weight) * stop_prediction_loss + self.reconstruction_loss_weight * recons_loss + position_loss

        return final_loss, recons_loss, stop_prediction_loss, position_loss
