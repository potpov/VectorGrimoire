import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import List
import pydiffvg

# code taken from Im2Vec and adapted
class SimpleVectorDecoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 paths: int = 4,
                 radius: int = 3,
                 render_size: int = 128,
                 **kwargs) -> None:
        super(SimpleVectorDecoder, self).__init__()

        self.latent_dim = latent_dim

        self.curves = paths
        self.number_of_points = self.curves * 3

        self.loss_fn = F.mse_loss

        self.circle_rad = radius
        self.render_size = render_size

        sample_rate = 1
        angles = torch.arange(0, self.number_of_points, dtype=torch.float32) *6.28319/ self.number_of_points
        self.id = self.sample_circle(self.circle_rad, angles, sample_rate)[:,:]
        self.base_control_features = torch.tensor([[1,0],[0,1],[0,1]], dtype=torch.float32, requires_grad=False)

        self.decode_transform = lambda x: x.permute(0, 2, 1)
        num_one_hot = self.base_control_features.shape[1]
        fused_latent_dim = latent_dim + num_one_hot + (sample_rate*2)


        self.point_predictor = nn.Sequential(
            nn.Conv1d(fused_latent_dim, fused_latent_dim, kernel_size=3, padding=1, padding_mode='circular', stride=1, dilation=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fused_latent_dim),
            nn.Conv1d(fused_latent_dim, fused_latent_dim, kernel_size=3, padding=1, padding_mode='circular', stride=1, dilation=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fused_latent_dim),
            nn.Conv1d(fused_latent_dim, fused_latent_dim, kernel_size=3, padding=1, padding_mode='circular', stride=1, dilation=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fused_latent_dim),
            nn.Conv1d(fused_latent_dim, fused_latent_dim, kernel_size=3, padding=1, padding_mode='circular', stride=1, dilation=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fused_latent_dim),
            nn.Conv1d(fused_latent_dim, 2, kernel_size=3, padding=1, padding_mode='circular', stride=1, dilation=1),
            nn.Sigmoid()  # bound spatial extent for image coordinates
        ) # TODO add better initialization

    def sample_circle(self, r, angles, sample_rate=10):
        pos = []
        for i in range(1, sample_rate+1):
            x = (torch.cos(angles*(sample_rate/i)) * r) + r
            y = (torch.sin(angles*(sample_rate/i)) * r) + r
            pos.append(x)
            pos.append(y)
        return torch.stack(pos, dim=-1)

    def raster(self, all_points: Tensor, color=[0,0,0,1], verbose=False, white_background=True, **kwargs):
        assert len(color) == 4
        device = all_points.device
        render_size = self.render_size
        bs = all_points.shape[0]
        if verbose:
            render_size = 768
        outputs = []
        all_points = all_points*render_size # brings point coordinates from [0,1] back to image scale

        num_ctrl_pts = torch.zeros(self.curves, dtype=torch.int32).to(device) + 2

        if(not isinstance(color, Tensor)):
            color = torch.tensor(color, dtype=torch.float32, requires_grad=True).to(device)

        for k in range(bs):
            # Get point parameters from network
            render = pydiffvg.RenderFunction.apply
            shapes = []
            shape_groups = []
            points = all_points[k].contiguous()#[self.sort_idx[k]] # .cpu()

            # had this issue a few times with the LR finder
            if(torch.isnan(points).any()):
                print(f"[WARNING] Found NaN values in points")
                points = torch.rand(points.shape).to(device)* 0.01

            # check if points are all collapsed, would throw DiffVG error..
            if(torch.all(points == 0.0)):
                print(f"[WARNING] Found all points to be at 0.0")
                noise_vec = torch.rand(points.shape).to(device)* 0.01
                points = points + noise_vec


            if verbose: # this creates color gradient for easier visual tracing in the rastered image
                end_color = color.to(device)
                start_color = torch.Tensor([0.0, 0.0, 0.0, 1.]).to(device)
                color_step_size = (end_color-start_color)/(self.curves)

                for i in range(self.curves): 
                    color_diff = color_step_size*i # the i creates the gradient -> different color for each segment
                    color = start_color + color_diff
                    color[3] = 0.9
                    # color = torch.tensor(color)
                    num_ctrl_pts = torch.zeros(1, dtype=torch.int32) + 2
                    if i*3 + 4 > self.curves * 3:
                        curve_points = torch.stack([points[i*3], points[i*3+1], points[i*3+2], points[0]])
                    else:
                        curve_points = points[i*3:i*3 + 4]
                    path = pydiffvg.Path(
                        num_control_points=num_ctrl_pts, points=curve_points,
                        is_closed=False, stroke_width=torch.tensor(2))
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([i]),
                        fill_color=None,
                        stroke_color=color)
                    shapes.append(path)
                    shape_groups.append(path_group)
                # add the points
                for i in range(self.curves * 3):
                    indicator_scale = 3
                    if i%3==0:
                        color = torch.tensor([1.,0.,1.,1.]) #fuchsia
                        shape = pydiffvg.Rect(p_min = points[i]-indicator_scale,
                                             p_max = points[i]+indicator_scale)
                        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.curves+i]),
                                                           fill_color=color)
                
                    else:
                        color = torch.tensor([0.,0.5,0.,1.]) #green
                        shape = pydiffvg.Circle(radius=torch.tensor(indicator_scale),
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

    def decode(self, z: Tensor, point_predictor=None) -> Tensor:
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
        
        base_control_features = self.base_control_features[None, :, :].repeat(bs, self.curves, 1 ).to(z.device) # I think this is the control variable c
        z_base = torch.cat([z, base_control_features], dim=-1)

        id = self.id[None, :, :].repeat(bs, 1, 1) # [ BS, curves * 3, 2], e.g. [32, 60, 2]
        fused_latent = torch.cat([z_base, id], dim=-1) # [bs, self.curves * 3, latent_dim + 2 (c) + 2 (p)]

        fused_latent = fused_latent.permute(0, 2, 1)

        # for compute_block in point_predictor[1:]:
        #     all_points = compute_block(all_points)

        all_points = point_predictor(fused_latent)
        all_points = all_points.permute(0, 2, 1)
        return all_points


    def forward(self, z: Tensor) -> List[Tensor]:
        all_points = self.decode(z)
        output = self.raster(all_points, white_background=True)
        return  [output, z, -1.0, -1.0]
    
