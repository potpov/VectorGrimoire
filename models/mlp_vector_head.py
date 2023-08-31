import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import List
import pydiffvg


class MLPVectorHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        paths: int = 4,
        render_size: int = 128,
        stroke_width: float = 1.0,
        **kwargs
    ) -> None:
        super(MLPVectorHead, self).__init__()

        self.latent_dim = latent_dim
        self.paths = paths
        self.render_size = render_size
        self.stroke_width = stroke_width

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(
                256, self.paths * 4 * 2 + 1
            ),  # 4 points per path, 2 coordinates per point, 1 stop prediction
            nn.Sigmoid(),
        )

    def raster(
        self,
        all_points: Tensor,
        color: Tensor = Tensor([0.0, 0.0, 0.0, 1.0]),
        verbose: bool = False,
        white_background: bool = True,
        **kwargs
    ) -> Tensor:
        """
        Rasterizes the predicted points of the cubic Bezier curves.

        Args:
            all_points: Tensor of shape (batch_size, self.paths * 4, 2)
            color: Tensor of shape (4)
            verbose: bool, currently unused
            white_background: bool, if True, the output will have a white background

        Returns:
            output: Tensor of shape (batch_size, 4, self.render_size, self.render_size)
        """
        device = all_points.device
        if verbose:
            render_size = 768
        else:
            render_size = self.render_size

        all_points = all_points * render_size
        num_ctrl_pts = torch.zeros(self.paths, dtype=torch.int32).to(device) + 2
        outputs = []

        for batch_idx in range(all_points.shape[0]):
            render = pydiffvg.RenderFunction.apply
            shapes = []
            shape_groups = []
            points = all_points[batch_idx].contiguous()

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts,
                points=points,
                is_closed=False,
                stroke_width=torch.tensor(self.stroke_width),
            )

            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=color,
                stroke_color=color,
            )
            shape_groups.append(path_group)
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                render_size, render_size, shapes, shape_groups
            )
            out = render(
                render_size,  # width
                render_size,  # height
                3,  # num_samples_x
                3,  # num_samples_y
                102,  # seed
                None,
                *scene_args
            )
            out = out.permute(2, 0, 1).view(4, render_size, render_size)
            outputs.append(out)
        output = torch.stack(outputs).to(all_points.device)
        if white_background:
            alpha = output[:, 3:4, :, :]
            output_white_bg = output[:, :3, :, :] * alpha + (1 - alpha)
            output = torch.cat([output_white_bg, alpha], dim=1)
        del num_ctrl_pts
        return output

    def forward(self, z: Tensor, verbose: bool = False) -> Tensor:
        out = self.linear(z)
        stop_predictions = out[:, -1]
        point_predictions = out[:, :-1]

        batch_size = point_predictions.shape[0]
        point_predictions = point_predictions.view(batch_size, self.paths * 4, 2)

        raster_images = self.raster(point_predictions, verbose=verbose)

        return raster_images, stop_predictions
