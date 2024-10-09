import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import List
import pydiffvg
import wandb


class CNNVectorHead(nn.Module):
    """
    The CNNVectorHead is similar to Im2Vec. It uses a unit cirlce as a filled shape as a basis and iteratively deforms that shape using a CNN.
    """

    def __init__(self,
                 latent_dim: int = 128,
                 segments: int = 4,
                 imsize: int = 64,
                 filled: bool = True,
                 pred_color: bool = False,
                 pred_alpha: bool = False,
                 pred_stroke_width: bool = False,
                 max_stroke_width: float = 10.0,
                 radius: float = 3., ):
        super(CNNVectorHead, self).__init__()

        self.latent_dim = latent_dim
        self.segments = segments
        self.imsize = imsize
        self.filled = filled
        self.pred_color = pred_color
        self.pred_alpha = pred_alpha
        self.pred_stroke_width = pred_stroke_width
        self.max_stroke_width = max_stroke_width
        self.radius = radius

        fused_latent_dim = latent_dim + 2 + 2  # 2 for point type (e.g. control point) and 2 for initial x-y-position on the circle
        self.num_points = self.segments * 3
        self.angles = torch.arange(0, self.num_points, dtype=torch.float32) * 6.28319 / self.num_points
        self.circle_point_positions = self.sample_circle(self.radius, self.angles)
        self.point_types = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.float32)

        # TODO this was 2 in the original Im2Vec code, but that would have doubled the number of points instead of keeping them the same
        padding = 1
        kernel_size = 3
        stride = 1
        dilation = 1

        self.point_predictor = nn.Sequential(
            nn.Conv1d(fused_latent_dim, fused_latent_dim * 2, kernel_size=kernel_size, padding=padding,
                      padding_mode='circular', stride=stride, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(fused_latent_dim * 2, fused_latent_dim * 2, kernel_size=kernel_size, padding=padding,
                      padding_mode='circular', stride=stride, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(fused_latent_dim * 2, fused_latent_dim * 2, kernel_size=kernel_size, padding=padding,
                      padding_mode='circular', stride=stride, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(fused_latent_dim * 2, fused_latent_dim * 2, kernel_size=kernel_size, padding=padding,
                      padding_mode='circular', stride=stride, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(fused_latent_dim * 2, fused_latent_dim * 2, kernel_size=kernel_size, padding=padding,
                      padding_mode='circular', stride=stride, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(fused_latent_dim * 2, 2, kernel_size=kernel_size, padding=padding, padding_mode='circular',
                      stride=stride, dilation=dilation),
            nn.Sigmoid()
        )

        # TODO do color and alpha and all that
        self.color_predictor = None
        if self.pred_color:
            self.color_predictor = nn.Sequential(
                nn.Linear(self.latent_dim, 3, bias=False),
                nn.Sigmoid()
            )

    def sample_circle(self, r, angles):
        """
        samples position on a circle of radius r, distances of positions are given by the angles, which are in [0, 2*pi]
        """
        pos = [(torch.cos(angles) * r), (torch.sin(angles) * r)]
        return torch.stack(pos, dim=-1)

    def raster(self, all_points, colors=None, white_background=True):

        render_size = self.imsize
        bs = all_points.shape[0]

        outputs = []
        all_paths = []
        pred_colors = []
        all_points = all_points * render_size
        num_ctrl_pts = torch.zeros(self.segments, dtype=torch.int32).to(all_points.device) + 2
        for k in range(bs):
            if colors is None:
                color = torch.tensor([0, 0, 0, 1]).to(all_points.device)
            else:
                color = colors[k].to(all_points.device)
            # Get point parameters from network
            render = pydiffvg.RenderFunction.apply
            points = all_points[k].contiguous()  # [self.sort_idx[k]] # .cpu()
            # print("points.shape: ", points.shape)
            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                is_closed=True)
            all_paths.append(path)

            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([0]),
                fill_color=color,
                stroke_color=color)

            # this for compositing
            pred_colors.append(color)

            scene_args = pydiffvg.RenderFunction.serialize_scene(render_size, render_size, [path], [path_group])
            out = render(render_size,  # width
                         render_size,  # height
                         3,  # num_samples_x
                         3,  # num_samples_y
                         102,  # seed
                         None,
                         *scene_args)
            out = out.permute(2, 0, 1).view(4, render_size, render_size)  # [:3]#.mean(0, keepdim=True)
            outputs.append(out)
        output = torch.stack(outputs).to(all_points.device)

        # map to [-1, 1]
        if white_background:
            alpha = output[:, 3:4, :, :]
            output_white_bg = output[:, :3, :, :] * alpha + (1 - alpha)
            output = torch.cat([output_white_bg, alpha], dim=1)
        del num_ctrl_pts, color
        return output, (all_paths, pred_colors)

    def forward(self, z, **kwargs):
        logging_dict = {}
        device = z.device
        bs = z.shape[0]

        if self.color_predictor:
            all_colors = self.color_predictor(z)
            all_colors = all_colors.view(bs, 3)
            # FIXME for now we add a new channel and set alpha to 1 manually
            all_colors = torch.cat([all_colors, torch.ones(bs, 1).to(device)], dim=-1)
        else:
            all_colors = None

        z = z[:, None, :].repeat(1, self.segments * 3, 1)

        # add point type information
        batched_point_types = self.point_types[None, :, :].repeat(bs, self.segments, 1).to(device)
        feats = torch.cat([z, batched_point_types], dim=-1)

        # add position on circle information
        positions = self.circle_point_positions[None, :, :].repeat(bs, 1, 1).to(device)
        feats = torch.cat([feats, positions], dim=-1)  # (bs, segments*3, latent_dim + 4)
        feats = feats.permute(0, 2, 1)  # (bs, latent_dim + 4, segments*3)

        all_points = self.point_predictor.forward(feats)  # (bs, 2, segments*3)
        all_points = all_points.permute(0, 2, 1)  # (bs, segments*3, 2)
        all_points = all_points.view(bs, self.num_points, 2)  # (bs, segments*3, 2)
        all_widths = torch.ones(bs, self.segments, 1)
        all_alphas = torch.ones(bs, self.segments, 1)

        output, (all_paths, all_groups) = self.raster(
            all_points,
            colors=all_colors
        )
        visual_attribute_dict = {
            "stroke_widths": all_widths,
            "alphas": all_alphas,
            "colors": all_colors
        }

        return [output, (all_paths, all_groups), all_points, visual_attribute_dict], logging_dict


class MLPVectorHead(nn.Module):
    """
    The MLPVectorHead is the default head of the VSQ where a fully-connected MLP predicts the position of each point and additional visual attributes like color.
    This class re-uses code from:
        - https://github.com/BachiLi/diffvg/blob/master/apps/generative_models/models.py#L17
        - https://github.com/BachiLi/diffvg/blob/master/apps/generative_models/rendering.py
        - https://github.com/BachiLi/diffvg/blob/master/apps/painterly_rendering.py
    """

    def __init__(self, latent_dim=128,
                 segments: int = 4,
                 imsize=32,
                 pred_color=False,
                 alpha_prediction=False,
                 stroke_width_predictor: bool = True,
                 max_stroke_width: float = 10.0,
                 dropout: float = 0.0):
        super(MLPVectorHead, self).__init__()

        self.stroke_width = max_stroke_width
        self.min_stroke_width = 0.3
        self.imsize = imsize
        self.segments = segments
        self.latent_dim = latent_dim
        self.stroke_width_predictor = stroke_width_predictor
        self.dropout = dropout

        # 4 points bezier with n_segments -> 3*n_segments + 1 points
        self.point_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SELU(),
            # nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SELU(),
            # nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, 2 * (self.segments * 3 + 1)),
            nn.Sigmoid()  # bound spatial extent
        )

        self.stroke_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SELU(),
            # nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, 1, bias=False),
            nn.Sigmoid()
        )

        self.alpha_predictor = None
        if alpha_prediction:
            self.alpha_predictor = nn.Sequential(
                nn.Linear(self.latent_dim, 1, bias=False),
                nn.Sigmoid()
            )

        self.color_predictor = None
        if isinstance(pred_color, bool) and pred_color:
            print("[INFOOO] color predictor enabled++++")
            self.color_predictor = nn.Sequential(
                # nn.Linear(self.latent_dim, self.latent_dim),
                # nn.SELU(),
                # nn.Dropout(self.dropout),
                # nn.Linear(self.latent_dim, self.latent_dim),
                # nn.SELU(),
                # nn.Dropout(self.dropout),
                nn.Linear(self.latent_dim, 3, bias=False),
                nn.Sigmoid()
            )

    def forward(self, z, primitive: str = "cubic", **kwargs):
        """
        outputs [output, scenes, all_points, visual_attribute_dict], logging_dict
        """
        logging_dict = {}
        bs = z.shape[0]

        feats = z
        all_points = self.point_predictor(feats)

        all_width = self.stroke_predictor(feats)
        if self.stroke_width_predictor:
            all_widths = self.stroke_predictor(feats) * self.stroke_width
        else:
            # we block the stroke width to 4.7 which globally means 0.7
            # a more proper way to compute this would be checking the longest stroke in the dataset before-hand
            all_widths = torch.ones_like(all_width) * 4.7

        all_widths = torch.max(all_widths,
                               torch.ones_like(all_widths) * self.min_stroke_width)  # enforce min stroke width

        # logging_dict["stroke_width"] = wandb.Histogram(all_widths.detach().cpu().flatten())
        logging_dict["mean_stroke_width"] = all_widths.detach().mean()

        if self.color_predictor:
            all_colors = self.color_predictor(feats)
            all_colors = all_colors.view(bs, 1, 3)
        else:
            all_colors = None

        if self.alpha_predictor:
            all_alphas = self.alpha_predictor(feats)
        else:
            all_alphas = torch.ones(all_widths.shape, device=all_widths.device)

        # min_width = self.stroke_width[0]
        # max_width = self.stroke_width[1]
        # all_widths = (max_width - min_width) * all_widths + min_width

        all_points = all_points.view(bs, 1, self.segments * 3 + 1, 2)

        output, scenes = self.bezier_render(
            all_points,
            all_widths,
            all_alphas,
            colors=all_colors,
            canvas_size=self.imsize,
            primitive=primitive
        )

        visual_attribute_dict = {
            "stroke_widths": all_widths,
            "alphas": all_alphas,
            "colors": all_colors
        }

        # map to [-1, 1]
        # output = output*2.0 - 1.0
        return [output, scenes, all_points, visual_attribute_dict], logging_dict

    def render(self,
               canvas_width,
               canvas_height,
               shapes,
               shape_groups,
               samples=2,
               seed=42):

        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)
        img = _render(canvas_width, canvas_height, samples, samples,
                      seed,  # seed
                      None,  # background image
                      *scene_args)
        return img

    def bezier_render(self, all_points: Tensor, all_widths: Tensor, all_alphas: Tensor,
                      canvas_size=32, primitive: str = "cubic", colors=None, white_background=True):
        device = all_points.device

        # all_points = 0.5*(all_points + 1.0) * canvas_size
        all_points = all_points * canvas_size

        eps = 1e-4
        all_points = all_points + eps * torch.randn_like(all_points, device=device)

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
                elif primitive == "quadratic":
                    num_ctrl_pts = torch.zeros(num_segments, dtype=torch.int32) + 1
                    points = points[[0, 1, 3]]
                else:
                    raise NotImplementedError(f"Primitive {primitive} not implemented")
                width = all_widths[batch, p]
                alpha = all_alphas[batch, p]
                if colors is not None:
                    color = colors[batch, p]
                else:
                    color = torch.zeros(3, device=device)

                color = torch.cat([color, alpha.view(1, )])

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
            raster = self.render(canvas_size, canvas_size, shapes, shape_groups,
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
            # output[batch] = torch.concat([image, raster[3:4]], dim=0)
            output[batch] = raster

        output = output.to(device)

        if white_background:
            alpha = output[:, 3:4, :, :]
            output_white_bg = output[:, :3, :, :] * alpha + (1 - alpha)
            output = torch.cat([output_white_bg, alpha], dim=1)

        return output, scenes
