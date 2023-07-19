import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from thesis.models import BaseVAE
from thesis.models.layers.improved_transformer import TransformerDecoderLayerGlobalImproved
from thesis.models.layers.transformer import TransformerDecoder
from thesis.models.vector_decoder import VectorDecoder
import wandb
import pydiffvg
import numpy as np
from typing import List

def make_tensor(x, grad=False):
    x = torch.tensor(x, dtype=torch.float32)
    x.requires_grad = grad
    return x

class LabelEmbedding(nn.Module):
    def __init__(self, n_labels, dim_label):
        super().__init__()

        self.label_embedding = nn.Embedding(n_labels, dim_label)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.label_embedding.weight, mode="fan_in")

    def forward(self, label):
        src = self.label_embedding(label)
        return src

class PositionalEncodingLUT(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncodingLUT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)

        self.pos_embed = nn.Embedding(max_len, d_model)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)
    
class ConstEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()

        self.seq_len = seq_len # = T

        self.d_model = d_model

        self.PE = PositionalEncodingLUT(d_model, max_len=seq_len)

    def forward(self, z):
        # N = z.size(1)
        bs = z.size(0)
        # src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        src = self.PE(z.new_zeros(self.seq_len, bs, self.d_model))
        return src
    
class PrefixEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        self.embedding = nn.Embedding(seq_len, d_model)
        nn.init.kaiming_normal_(self.embedding.weight, mode="fan_in")

    def forward(self, z:Tensor):
        bs = z.shape[0]


    

class LatentTransformer(nn.Module):
    def __init__(self, n_embedds = 15, dim_model = 256, n_heads = 8, n_layers_decode = 4, dim_FF = 512, dropout = 0.1, dim_z = 256, n_labels = 100, dim_label = 64, label_condition = False, **kwargs):
        super().__init__()

        self.label_condition = label_condition
        dim_label = dim_label if self.label_condition else None
        if(self.label_condition):
            self.label_embedding = LabelEmbedding(n_labels, dim_label)

        self.embedding = ConstEmbedding(n_embedds, dim_model)

        decoder_layer = TransformerDecoderLayerGlobalImproved(dim_model, dim_z, n_heads, dim_FF, dropout)
        decoder_norm = nn.LayerNorm(dim_model)
        self.decoder = TransformerDecoder(decoder_layer, n_layers_decode, decoder_norm)

        self.fcn = nn.Linear(dim_model, dim_z)
    
    def forward(self, z, label = None):
        """
        requires z to be of shape: []
        """
        # N = z.size(2)
        l = self.label_embedding(label).unsqueeze(0) if self.label_condition else None

        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None, memory2=l)

        return out

class CNNVectorDecoder(VectorDecoder):

    def __init__(self,
                 latent_dim: int,
                 loss_fn: str = 'MSE',
                 imsize: int = 128,
                 paths: int = 4,
                 b_w = True,
                 wandb_logging = None,
                 **kwargs) -> None:
        super(CNNVectorDecoder, self).__init__(latent_dim,
                                               loss_fn,
                                               paths,
                                               wandb_logging,
                                               **kwargs)
        self.imsize = imsize
        def get_computational_unit(in_channels, out_channels, unit):
            if unit == 'conv':
                return nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular', stride=1,
                                 dilation=1)
            else:
                return nn.Linear(in_channels, out_channels)

        T = kwargs['T']
        self.T = T
        print(f"using {T} paths")
        if(b_w):
            if(kwargs["optimize_colors"]):
                self.colors = torch.nn.Parameter(torch.Tensor([0.,0.,0.,1.]).unsqueeze(0).repeat(T,1))
            else:
                self.colors = torch.Tensor([[0., 0., 0., 1.] for _ in range(T)])
        else:
            # self.colors = torch.nn.Parameter([get_random_color() for _ in range(T)], requires_grad=kwargs["optimize_colors"])
            self.colors = torch.nn.Parameter(torch.Tensor([0.5,0.5,0.5,1.]).unsqueeze(0).repeat(T,1), requires_grad=kwargs["optimize_colors"])

        self.composite_fn = self.hard_composite
        if kwargs['composite_fn'] == 'soft':
            print('Using Differential Compositing')
            self.composite_fn = self.soft_composite
        self.divide_shape = nn.Sequential(
            nn.ReLU(),  # bound spatial extent
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # nn.ReLU(),  # bound spatial extent
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # nn.ReLU(),  # bound spatial extent
        )
        self.final_shape_latent = nn.Sequential(
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
        )
        self.z_order = nn.Sequential(
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # nn.ReLU(),  # bound spatial extent
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, 1, 'mlp'),
        )
        layer_id = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.register_buffer('layer_id', layer_id)

        # Logging stuff
        self.wandb_logging = wandb_logging # None if not logging

    def forward(self, transformed_z: Tensor, **kwargs) -> List[Tensor]:
        output, control_loss = self.decode_and_composite(transformed_z, verbose=False, return_overlap_loss=True, **kwargs)
        return [output, transformed_z, control_loss]

    def hard_composite(self, **kwargs):
        layers = kwargs['layers']
        n = len(layers)
        alpha = (1 - layers[n - 1][:, 3:4, :, :])
        rgb = layers[n - 1][:, :3] * layers[n - 1][:, 3:4, :, :]
        for i in reversed(range(n-1)):
            rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * alpha
            alpha = (1-layers[i][:, 3:4, :, :]) * alpha
        rgb = rgb + alpha
        return rgb

    def soft_composite(self, **kwargs):
        layers = kwargs['layers']
        z_layers = kwargs['z_layers']
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


    def soft_composite_W_bg(self, **kwargs):
        layers = kwargs['layers']
        z_layers = kwargs['z_layers']
        n = len(layers)

        sum_alpha = layers[0][:, 3:4, :, :] * z_layers[0]
        for i in range(1, n):
            sum_alpha = sum_alpha + layers[i][:, 3:4, :, :] * z_layers[i]
        sum_alpha = sum_alpha + 600

        inv_mask = 600 / sum_alpha

        rgb = layers[0][:, :3] * layers[0][:, 3:4, :, :] * z_layers[0] / sum_alpha
        for i in range(1, n):
            rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * z_layers[i] / sum_alpha
        rgb = rgb + inv_mask
        return rgb
    
    def manhattan_dist(self, triplet:Tensor) -> float:
        """calculates the summed manhattan distance between the three points of a cubic bezier curve.

        Parameters:
        self: self
        triplet (Tensor): contains the coordinates of a cubic bezier curve in the pixel space
        """
        first = abs(triplet[1][0] - triplet[0][0]) + abs(triplet[1][1] - triplet[0][1])
        second = abs(triplet[2][0] - triplet[1][0]) + abs(triplet[2][1] - triplet[1][1])

        return first.item()+second.item()

    def log_path_lengths(self, all_points:Tensor, current_shape_idx:int):
        """
        logs the path lengths of each full shape (consisting of individual bezier curves) in a batch.
        """
        logging_dict = {}
        for batch_idx in range(all_points.shape[0]):
            total_length = 0
            for curve_idx in range(0, self.curves*3, 3):
                try:
                    total_length += self.manhattan_dist(all_points[batch_idx][curve_idx*3:curve_idx*3+3])
                except:
                    pass
            logging_dict[f"shapeidx_{current_shape_idx}_batchidx_{batch_idx}"] = total_length
            break
        wandb.log(logging_dict)

    def decode_and_composite(self, transformed_z: Tensor, return_overlap_loss=False, **kwargs):
        # bs = z.shape[0]
        layers = []
        loss = 0
        # z_rnn_input = z[None, :, :].repeat(n, 1, 1)  # [len, batch size, emb dim], copies the latent code for each shape of the composition
        # outputs, hidden = self.rnn(z_rnn_input)
        # outputs = outputs.permute(1, 0, 2)  # [batch size, len, emb dim]
        # outputs = outputs[:, :, :self.latent_dim] + outputs[:, :, self.latent_dim:] # aggregate outputs of both RNNs
        z_layers = []
        for i in range(self.T):
            # this handles T=1
            if(transformed_z.dim() > 2):
                current_z = transformed_z[:,i,:]
            else:
                current_z = transformed_z
            # shape_output = self.divide_shape(transformed_z[:, i, :]) # [bs, latent_size]
            # shape_latent = self.final_shape_latent(transformed_z) # [bs, latent_size]
            all_points = self.decode(current_z)#, point_predictor=self.point_predictor[i])
            if("log_path_length" in kwargs.keys() and self.wandb_logging):
                if(kwargs["log_path_length"]):
                    self.log_path_lengths(all_points*self.imsize, current_shape_idx=i)

            # print(torch.isfinite(all_points).all())
            # import pdb; pdb.set_trace()
            if("verbose" in kwargs):
                if(kwargs["verbose"]):
                    gradient_end_colors = [np.array((0., 0., 1., 1)), np.array((0., 1., 0., 1)), np.array((1., 0., 0., 1))]
                    layer = self.raster(all_points, gradient_end_colors[i], verbose=True, white_background=False)
                else:
                    layer = self.raster(all_points, self.colors[i], verbose=False, white_background=False)
            else:
                layer = self.raster(all_points, self.colors[i], white_background=False)

            z_pred = self.z_order(current_z)
            layers.append(layer)
            z_layers.append(torch.exp(z_pred[:, :, None, None]))
            if return_overlap_loss:
                loss = loss + self.control_polygon_distance(all_points)
        if(self.wandb_logging):
            wandb.log({"overlap_loss":loss})
        output = self.composite_fn(layers = layers, z_layers=z_layers)

        if return_overlap_loss:
        #     overlap_alpha = layers[1][:, 3:4, :, :] + layers[2][:, 3:4, :, :]
        #     loss = F.relu(overlap_alpha - 1).mean()
            return output, loss
        return output

    def generate(self, z: Tensor, **kwargs) -> Tensor:
        """
        Given an input latent z, generates the reconstructed image
        :param z: (Tensor) [B x T x DIM]
        :return: (Tensor) [B x C x H x W]
        """
        # mu, log_var = self.encode(x)
        # z = self.reparameterize(mu, log_var)
        output = self.decode_and_composite(z, **kwargs) # might want to use verbose keyword here
        return output  # [:, :3]

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
            output = self.decode_and_composite(z, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations

    def interpolate_mini(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = self.interpolate_vectors(mu[0], mu[1], 10)
        output = self.decode_and_composite(z, verbose=kwargs['verbose'])
        return output

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
            output = self.decode_and_composite(z, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations

   

    def visualize_sampling(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(7, 25):
            self.redo_features(i)
            output = self.decode_and_composite(mu, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations


    def save(self, all_points, save_dir, name, verbose=False, white_background=True):
        # note that this if for a single shape and bs dimension should have multiple curves
        # print('1:', process.memory_info().rss*1e-6)
        render_size = self.imsize
        bs = all_points.shape[0]
        if verbose:
            render_size = render_size*2
        all_points = all_points*render_size
        num_ctrl_pts = torch.zeros(self.curves, dtype=torch.int32) + 2

        shapes = []
        shape_groups = []
        for k in range(bs):
            # Get point parameters from network
            color = make_tensor(color[k])
            points = all_points[k].cpu().contiguous()#[self.sort_idx[k]]

            if verbose:
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
                for i in range(self.curves * 3):
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
        pydiffvg.save_svg(f"{save_dir}{name}/{name}.svg",
                          self.imsize, self.imsize, shapes, shape_groups)

class VAEctorGen(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        dim_z: int,
        T:int,
        hidden_dims: list = None,
        **kwargs
    ) -> None:
        
        super(VAEctorGen, self).__init__()

        self.dim_z = dim_z

        # CNN encoder, calculate with 128 img size
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Mean and variance of distribution
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, self.dim_z)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, self.dim_z)

        self._init_embeddings()

        # Build Decoder
        # self.mapping = nn.Linear(dim_z, kwargs["dim_model"])

        ## Transformer for latent code
        self.transformer = LatentTransformer(n_embedds=T, dim_z=dim_z, **kwargs)

        ## CNN deformation network
        self.decoder = CNNVectorDecoder(in_channels=in_channels, latent_dim=dim_z, T=T, **kwargs)
        

    def _init_embeddings(self):
        nn.init.normal_(self.fc_mu.weight, std=0.001)
        nn.init.constant_(self.fc_mu.bias, 0)
        nn.init.normal_(self.fc_var.weight, std=0.001)
        nn.init.constant_(self.fc_var.bias, 0)


    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        
    def encode(self, imgs:Tensor):
        # encode image through encoder
        encoded_images = self.encoder(imgs)
        encoded_images = encoded_images.view(encoded_images.size(0), -1) # result [Batch, Features]
        return encoded_images

    def forward(
        self,
        img: Tensor,
        label:Tensor = None,
        **kwargs
    ):
        encoded_images = self.encode(img)
        # get mean and var
        mu = self.fc_mu(encoded_images)
        var = self.fc_var.forward(encoded_images)

        # reparameterization trick
        z = self.reparameterize(mu, var)

        #mapped_z = self.mapping(z)
        mapped_z = z

        # apply transformer
        transformed_z = self.transformer.forward(mapped_z, label=label)

        transformed_z = transformed_z.squeeze(0)

        # adapt to batch first - [B x T x LatentDim]
        transformed_z = transformed_z.permute(1,0,2)

        # decode latent codes
        output, transformed_z, control_loss = self.decoder.forward(transformed_z, **kwargs)
        
        return output, img, mu, var, control_loss # required for loss function

    def generate(self, test_input, labels, **kwargs):
        # results = self.forward(test_input, labels, **kwargs)
        encoded_input = self.encode(test_input)
        mu = self.fc_mu(encoded_input)
        var = self.fc_var.forward(encoded_input) # TODO check if this is var or log_var

        z = self.reparameterize(mu, var)

        transformed_z = self.transformer.forward(z, label=labels)

        transformed_z = transformed_z.squeeze(0)

        # adapt to batch first - [B x T x LatentDim]
        transformed_z = transformed_z.permute(1,0,2)

        return self.decoder.generate(transformed_z, labels=labels, **kwargs)
        # return results[0]
    
    def sample(self, num_samples: int, current_device:int, label = None, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.dim_z).to(current_device)
        
        # apply transformer
        transformed_z = self.transformer.forward(z, label=label)

        transformed_z = transformed_z.squeeze(0)

        # adapt to batch first - [B x T x LatentDim]
        transformed_z = transformed_z.permute(1,0,2)

        # decode latent codes
        output, _, _ = self.decoder.forward(transformed_z, **kwargs)
        return output

    def loss_function(self,
                      recons:Tensor,
                      input:Tensor,
                      mu,
                      log_var,
                      other_losses:int = 0,
                      **kwargs) -> dict:

        return self.decoder.loss_function(recons, input, mu, log_var, other_losses, **kwargs)