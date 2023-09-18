import os
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import List
import pydiffvg
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
# from torchmetrics.functional.image import learned_perceptual_image_patch_similarity - functional interface does not work on GPU I think
from PIL import Image, ImageDraw, ImageFont
import imageio
from io import BytesIO
import argparse
render = pydiffvg.RenderFunction.apply

def sample_circle(num_points: int, radius: float = 1.,):
    """
    Args:
        - num_points, int: how many points to sample
        - radius, float: radius of the circle, default: 1.0
    """
    pos = []
    angles = torch.arange(0, 2*torch.pi, 2*torch.pi / num_points)
    for i in range(1, 2):
        x = ((torch.cos(angles*(1/i)) * radius) + radius) / 2
        y = ((torch.sin(angles*(1/i)) * radius) + radius) / 2
        pos.append(x)
        pos.append(y)
    return torch.stack(pos, dim=-1)

def get_points(primitive:str, num_segments:int, mode:str):
    """
    Gets initial point positions between [0, 1] in a (num_points, 2) float32 tensor

    Args:
        - primitive, str: what svg primitive to use, currently supported "line" and "cubic" for paths
        - num_segments, int: how many segments for a single path?
        - mode, str: init mode for the points, currently supported "middle", "waves", "circle"

    Returns:
        - points in a (num_points, 2) float tensor, [0, 1] range
    """
    assert num_segments >= 1
    assert primitive in ["line", "cubic"]
    assert mode in ["middle", "waves", "circle", "random"]

    total_points = None
    
    if primitive == "line":
        total_points = 1 + num_segments
    elif primitive == "cubic":
        total_points = 1 + num_segments*3

    points = torch.zeros((total_points, 2)).type(torch.float32)
    left_right_padding = 0.2

    if mode == "middle":
        points = points + 0.5
        epsilon = torch.randn((total_points, 2)) * 0.0001
        points = points + epsilon  # required to not run into diffvg length 0 runtime error
    elif mode == "waves":
        dist_between_points = 1 / total_points
        if primitive == "line":
            for i in range(0, total_points, 2):
                points[i, 0] = left_right_padding
                points[i, 1] = i * dist_between_points
                if i+1 < len(points):
                    points[i+1, 0] = 1.0 - left_right_padding
                    points[i+1, 1] = (i+1) * dist_between_points
        elif primitive == "cubic":
            for i in range(0, total_points, 3):
                points[i, 0] = left_right_padding
                points[i, 1] = i * dist_between_points
                points[i+1 : i+3, 0] = 1.0 - left_right_padding
                points[i+1 : i+3, 1] = (i+1) * dist_between_points + 0.5 * dist_between_points
    elif mode == "circle":
        # TODO this is currently not quite correct as the padding is not applied to half of the borders (still starts at 0.0;0.0)
        points = sample_circle(total_points, radius=1. - left_right_padding)
    elif mode == "random":
        points = torch.rand((total_points, 2))

    return points

def get_initial_component(primitive = "cubic", 
                          resolution = 128, 
                          mode: str = "middle", 
                          num_segments:int = 2, 
                          filled:bool = False, 
                          stroke_width:float = 1.0, 
                          grad = True, 
                          shapes: list = None,  # this will be extended if not None, used for multiple paths
                          shape_groups: list = None):  # this will be extended, too

    points = get_points(primitive, num_segments, mode) * resolution
    points.requires_grad = grad
    assert points.is_leaf, "points is not leaf node"

    color = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=grad)
    if filled:
        fill_color = torch.tensor([0.0, 0.0, 0.0, 1.], requires_grad=grad)
    else:
        fill_color = None
    stroke_width = torch.tensor(stroke_width, dtype=torch.float32, requires_grad=grad)

    if shapes is None:
        shapes = []
    else:
        shapes = shapes
    if shape_groups is None:
        shape_groups = []
    else:
        shape_groups = shape_groups

    points = points.contiguous()

    if primitive == "cubic":
        num_ctrl_pts = torch.zeros(num_segments, dtype=torch.int32) + 2
    else:
        num_ctrl_pts = torch.zeros(num_segments, dtype=torch.int32)

    path = pydiffvg.Path(
                num_control_points=num_ctrl_pts,
                points=points,
                is_closed=False,
                stroke_width=stroke_width,
            )

    shapes.append(path)

    path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    # fill_color=fill_color, TODO ohfa
                    fill_color=color,
                    stroke_color=color,
                )

    shape_groups.append(path_group)

    return points, num_ctrl_pts, color, fill_color, stroke_width, shapes, shape_groups

def render_scene(shapes, shape_groups, resolution = 128, format = "CHW"):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
                resolution, resolution, shapes, shape_groups
            )

    out = render(   resolution,  # width
                    resolution,  # height
                    5,  # num_samples_x for monte carlo sampling
                    5,  # num_samples_y for monte carlo sampling
                    102,  # seed
                    None,
                    *scene_args
                )
    
    if format == "CHW":
        out = out.permute(2, 0, 1)
        alpha = out[3:4, :, :]
        output_white_bg = out[:3, :, :] * alpha + (1 - alpha)
        output = torch.cat([output_white_bg, alpha], dim=0)
        output = output
        return output
    else:
        raise ValueError("please provide supported format")

def make_target(resolution, path:str = "/home/mfeuerpfeil/master/thesis/datasets/mnist_png/training/2/10024.png", invert=True):
    """
    return target image in 4-channel RGBA format with C x W x H, normalized -> values between 0 and 1
    """
    target = Image.open(path).convert("RGB")
    if invert:
        target = transforms.RandomInvert(p=1.)(target)
    target = transforms.Resize(resolution)(target)
    target = target.convert("RGBA")
    target = transforms.ToTensor()(target)
    target[3,:,:] = 1.
    return target

def get_verbose_components(points:Tensor, primitive:str, end_color:Tensor = torch.tensor([1.0, 0.0, 0.0, 1.]), verbose=True):
    """"
    turns already scaled points into verbose shapes and shape groups
    """
    if primitive == "cubic":
        num_segments = len(points)//3
    else:
        num_segments = len(points) - 1
    shapes = []
    shape_groups = []
    points = points.detach()
    start_color = torch.tensor([0.0, 0.0, 0.0, 1.])
    color_step_size = (end_color-start_color)/(num_segments)
    all_colors = (list(plt.get_cmap("tab20b").colors))
    

    # draw the real bezier paths
    for i in range(num_segments):
        color_diff = color_step_size*i # the i creates the gradient -> different color for each segment
        color = start_color + color_diff
        color[3] = 0.9
        # color = torch.tensor(color)
        if primitive == "cubic":
            num_ctrl_pts = torch.zeros(1, dtype=torch.int32) + 2
        else:
            num_ctrl_pts = torch.zeros(1, dtype=torch.int32)

        # # check circular closed condition
        # if i*3 + 4 > num_segments * 3:
        #     single_path_points = torch.stack([points[i*3], points[i*3+1], points[i*3+2], points[0]])
        # else:
        if primitive == "cubic":
            if verbose:
                print(f"Looking at segment from {i*3} to {i*3+4}")
            single_path_points = points[i*3:i*3 + 4]
        else:
            if verbose:
                print(f"Looking at segment from {i} to {i+1}")
            single_path_points = points[i:i+1]

        path = pydiffvg.Path(
            num_control_points=num_ctrl_pts, points=single_path_points,
            is_closed=False, stroke_width=torch.tensor(2))
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=None,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.]))
            # stroke_color=color)
        shapes.append(path)
        shape_groups.append(path_group)

    # print(f"[INFO] got {len(shapes)} paths.")
    # add the actual points
    for i, point in enumerate(points):
        try:
            curr_color = torch.cat([torch.tensor(all_colors[i]), torch.tensor([1.])])
        except Exception as e:
            curr_color = torch.tensor([1.,0.,0.,1.])
        indicator_scale = 3

        if primitive == "cubic":
            # first point is always an anchor point
            if i%3==0:
                # color = torch.tensor([1.,0.,1.,1.]) #fuchsia
                shape = pydiffvg.Rect(p_min = point-indicator_scale,
                                        p_max = point+indicator_scale)
                group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([num_segments+i]),
                                                    fill_color=curr_color)
            # all other points are control points
            else:
                # color = torch.tensor([0.,0.5,0.,1.]) #green
                shape = pydiffvg.Circle(radius=torch.tensor(indicator_scale),
                                            center=point)
                group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([num_segments+i]),
                                                    fill_color=curr_color)
        else:
            shape = pydiffvg.Rect(p_min = point-indicator_scale,
                                    p_max = point+indicator_scale)
            group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([num_segments+i]),
                                        fill_color=curr_color)

        shapes.append(shape)
        shape_groups.append(group)

    return shapes, shape_groups

def optimize(points_vars,
             stroke_width_vars,
             color_vars,
             optimizable_shapes,
             optimizable_groups,
             target,
             resolution,
             primitive:str,
             loss_name = "MSE", # MSE or LPIPS
             num_iter:int = 200,
             max_width:float = 10.,
             loss_scaling:float = 1.,
             verbose = True):
    step_images = None
    points_grad = []
    losses = []

    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)
    # points_optim = torch.optim.SGD(points_vars, lr=1.0)
    # width_optim = torch.optim.SGD(stroke_width_vars, lr=0.1)
    # color_optim = torch.optim.SGD(color_vars, lr=0.01)

    loss_fn = None
    lpips = None

    if loss_name.lower() == "mse":
        loss_fn = F.mse_loss
    elif loss_name.lower() == "lpips":
        lpips = LearnedPerceptualImagePatchSimilarity("squeeze", "mean", normalize=True)
        loss_fn = lpips.forward
    elif loss_name.lower() == "mix":
        lpips = LearnedPerceptualImagePatchSimilarity("squeeze", "mean", normalize=True)
        def mix(img1, img2):
            loss = lpips.forward(img1, img2) + F.mse_loss(img1, img2)
            return loss
        loss_fn = mix
    else:
        raise ValueError("Please choose a valid loss function")
    if verbose:
        print("Start optimizing.")
    for t in range(num_iter):
        if verbose and t % 10 == 0:
            print('iteration:', t)
        # print(f"current stroke: {stroke_width.item()}")
        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()

        # for path in shapes:
            # print(path.points.requires_grad, path.stroke_width.requires_grad)

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            resolution, resolution, optimizable_shapes, optimizable_groups)
        
        img = render(resolution, # width
                resolution, # height
                2,   # num_samples_x
                2,   # num_samples_y
                0,   # seed
                None,
                *scene_args)
        
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        
        # Save the intermediate render.
        # pydiffvg.imwrite(img.cpu(), os.path.join(BASE_PATH, 'results/painterly_rendering/iter_{}.png'.format(t)), gamma=gamma)
        img = img[:, :, :3]
        
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        if step_images is None:
            step_images = img.detach().clone().cpu()
        else:
            # TODO this is not efficient copying the tensor every iteration
            step_images = torch.cat((step_images, img.detach().clone().cpu()), dim=0)

        if target.size(0) == 4:
            target = target[:3, :, :]

        # target = torch.zeros((3, resolution, resolution))

        # print(img.shape, target.shape)
        target = target.to(img.device)
        if lpips is not None:
            lpips = lpips.to(img.device)
        # loss = (img - target).pow(2).mean() * loss_scaling
        if lpips is not None and target.dim() < 4:
            target = target.unsqueeze(dim=0)
        loss = loss_fn(img, target) * loss_scaling
        losses.append(loss.detach().item())
        if verbose and t % 10 == 0:
            print('render loss:', np.round(loss.item(), 4))

        # Backpropagate the gradients.
        loss.backward()

        points_grad.append(points_vars[0].grad.clone().detach().cpu())

        # Take a gradient descent step.
        points_optim.step()
        if len(stroke_width_vars) > 0:
            width_optim.step()
            # print("BACKWARD width")
        color_optim.step()  # TODO currently not optimizing color, add as parameter
        if len(stroke_width_vars) > 0:
            for path in optimizable_shapes:
                path.stroke_width.data.clamp_(1.0, max_width)

        for group in optimizable_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)
    verbose_scaling = 4.0
    shapes, groups = get_verbose_components(points_vars[0]*verbose_scaling, primitive)
    final_verbose_output = render_scene(shapes, groups, resolution = int(resolution*verbose_scaling))
    return step_images, final_verbose_output, losses, points_grad

def add_loss_to_optimization_process_image(grid:Tensor, losses:List[float]):
    image_tensor = grid[:3,:,:]
    original_width = image_tensor.shape[2]
    original_heigth = image_tensor.shape[1]

    fig, ax = plt.subplots(figsize=(original_width/1000, original_heigth/1000))
    ax.plot(losses, linewidth=3, label="loss")
    ax.grid(True)
    ax.set_xlabel("timestep")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)

    # Load the line plot image as a PIL Image
    line_plot_image = Image.open(buffer)

    line_plot_tensor = torch.tensor(np.array(line_plot_image).transpose((2, 0, 1)))/255  # Convert HWC to CHW
    # save_image(line_plot_tensor, "images/line_plot_tensor.png")
    # Resize the line plot tensor to match the width of the original image
    original_width = image_tensor.shape[2]
    line_plot_tensor:Tensor = torch.nn.functional.interpolate(line_plot_tensor.unsqueeze(0), size=(line_plot_tensor.shape[1], original_width), mode='bilinear', align_corners=False).squeeze(0)

    # Determine the size of the final extended image
    extended_height = image_tensor.shape[1] + line_plot_tensor.shape[1]

    # Create a new tensor with the extended height
    extended_image = torch.zeros(3, extended_height, original_width)

    # print(image_tensor.shape, line_plot_tensor.shape, extended_image.shape)

    # Copy the line plot tensor to the bottom of the extended image tensor
    extended_image[:3, image_tensor.shape[1]:, :] = line_plot_tensor[:3,:,:]

    # Copy the original image tensor to the top of the extended image tensor
    extended_image[:, :image_tensor.shape[1], :] = image_tensor[:3,:,:]

    return extended_image

def save_optimization_process_image(step_images:Tensor, final_verbose_output:Tensor, target:Tensor, path:str, title:str="Ooga booga", losses:List[float] = None):
    step_images = step_images.cpu()
    final_verbose_output = final_verbose_output.cpu()
    target = target.cpu()
    step_grid = make_grid(step_images, nrow=int(len(step_images)**0.5))  # C x H x W
    resizer = transforms.Resize(step_grid.shape[1:], antialias=True)
    resized_target = resizer(target)
    resized_final_verbose_output = resizer(final_verbose_output)[:3,:,:]
    final_grid = make_grid([step_grid, resized_target,resized_final_verbose_output], nrow=3)
    if losses is not None:
        final_grid = add_loss_to_optimization_process_image(final_grid, losses)
        plt.clf()
    plt.imshow(final_grid.permute(1,2,0))
    plt.title(title)
    plt.grid(False)
    # save_image(final_grid, path)
    plt.savefig(path, dpi=500)
    return final_grid

def load_npy_timeseries_image(path: str = "/scratch2/moritz_data/CausalMNISTpp/I10029_P9.npy"):
    raw_data = torch.from_numpy(np.load(path))
    assert raw_data.dim() == 3, "raw_data has wrong dimensions, expected T x W x H, NO CHANNEL"
    if raw_data.mean(dtype=torch.float32) > 1.:
        raw_data = raw_data/255.
    stacked_images = raw_data.unsqueeze(1).repeat(1,3,1,1)
    return stacked_images

def step_images_to_mp4(step_images:Tensor, output_path:str = "output.mp4", title:str="", fps:int = 24):
    """
    Input:
        step_images: Tensor of shape T x C x W x H
        output_path: path to save the mp4 to
        title: title to be displayed on each frame
        fps: frames per second
    
    Returns:
        None
    """
    assert fps > 0, "fps must be greater than 0"
    font = ImageFont.load_default()
    frames = []
    for i, timestep in enumerate(step_images):
        # Create a PIL Image from the numpy array (assuming RGB format)
        timestep = timestep.permute(1, 2, 0).cpu().numpy()*255
        timestep = timestep.astype(np.uint8)
        width, height, channels = timestep.shape
        image = Image.fromarray(timestep, mode='RGB')
        draw = ImageDraw.Draw(image)
        draw.text((10, 5), title, fill=(0, 0, 0), font=font)
        draw.text((10, 20), f"Step {i}", fill=(0, 0, 0), font=font)
        frames.append(image)
    imageio.mimsave(output_path, frames, fps=fps)

def main(args: argparse.Namespace):
    resolution: int = args.resolution
    num_iter: int = args.num_iter
    max_width: float = args.max_width

    # Load target image based on args
    target_path: str = args.target_image_path
    # TODO add the selection of alpha channel here, also with option for optimizatioin
    target = make_target(resolution, target_path, invert=False)[:3,:,:]

    output_path = args.output_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_grids = []
    all_losses = []

    grid_num_paths: List[int] = args.num_paths
    grid_segments: List[int] = args.segments
    grid_loss_scales: List[float] = args.loss_scales
    grid_stroke_widths: List[float] = args.stroke_widths
    grid_losses: List[str] = args.losses
    grid_modes: List[str] = args.modes
    grid_primitives: List[str] = args.primitives

    verbose: bool = args.verbose
    override: bool = args.override
    filled: bool = args.filled

    total_steps = len(grid_num_paths) * len(grid_segments) * len(grid_loss_scales) * len(grid_stroke_widths) * len(grid_losses) * len(grid_modes) * len(grid_primitives)
    curr_step = 0
    for num_paths in grid_num_paths:
        for primitive in grid_primitives:
            for mode in grid_modes:
                for num_segments in grid_segments:
                    for loss_scale in grid_loss_scales:
                        for initial_stroke_width in grid_stroke_widths:
                            for lossfn in grid_losses:
                                if verbose:
                                    print(f"{primitive}, {mode}, {num_segments}, {lossfn}")
                                
                                optimization_process_grid_path = os.path.join(output_path, f"optimization_process_{'filled' if filled else 'unfilled'}_{num_paths}_{primitive}s_{mode}_{lossfn.upper()}_{num_segments}_segments_{num_iter}_iters_{loss_scale}_loss_scale_{initial_stroke_width}_stroke.png")
                                optimization_video_path = os.path.join(output_path, f"optimization_video_{'filled' if filled else 'unfilled'}_{num_paths}_{primitive}s_{mode}_{num_segments}_segments_{lossfn.upper()}_{num_iter}_iterations_{loss_scale}_loss_scale_{initial_stroke_width}_stroke.mp4")
                                plot_title = f"{lossfn.upper()}, {num_segments} {primitive}s, Iters: {num_iter}, Mode: {mode}, {'filled' if filled else 'unfilled'}"

                                if os.path.exists(optimization_process_grid_path) and os.path.exists(optimization_video_path):
                                    if override:
                                        pass
                                    else:
                                        print("Skipping because already exists, all_grids will probably be incomplete. Use --override to override existing plots.")
                                        curr_step = curr_step+1  # TODO this can be better placed tbh
                                        continue
                                points_vars = []
                                stroke_width_vars = []
                                color_vars = []

                                shapes = []
                                shape_groups = []

                                for i in range(num_paths):
                                    points, num_ctrl_pts, color, fill_color, stroke_width, shapes, shape_groups = get_initial_component(primitive=primitive,
                                                                                                                                        grad = True,
                                                                                                                                        resolution=resolution,
                                                                                                                                        num_segments = num_segments,
                                                                                                                                        filled = filled,
                                                                                                                                        mode=mode,
                                                                                                                                        stroke_width=initial_stroke_width,
                                                                                                                                        shapes = shapes,
                                                                                                                                        shape_groups = shape_groups)
                                    points_vars.append(points)
                                    stroke_width_vars.append(stroke_width)
                                    color_vars.append(color)


                                # Optimize
                                step_images, final_verbose_output, losses, points_grad = optimize(points_vars,
                                                                                                stroke_width_vars,
                                                                                                color_vars,
                                                                                                shapes,
                                                                                                shape_groups,
                                                                                                target.to(points_vars[0].device),
                                                                                                resolution,
                                                                                                primitive,
                                                                                                num_iter = num_iter,
                                                                                                max_width = max_width,
                                                                                                loss_scaling=loss_scale,
                                                                                                loss_name = lossfn,
                                                                                                verbose = verbose)
                                grid = save_optimization_process_image(step_images,
                                                                        final_verbose_output,
                                                                        target,
                                                                        optimization_process_grid_path,
                                                                        title = plot_title,
                                                                        losses = losses)
                                final_output = step_images[-1]
                                save_image(final_output, optimization_process_grid_path.replace(".png", "_final_output.png"))
                                fps = int(len(step_images) / 20)  # this should ensure that the video is 20 seconds long no matter the number of steps
                                if verbose:
                                    print(f"Saving in {fps} fps")
                                step_images_to_mp4(step_images, 
                                                optimization_video_path, 
                                                title = f"{lossfn.upper()}, {num_segments} {primitive}, {mode} init",
                                                fps = fps)  
                                all_grids.append(grid)
                                all_losses.append(losses[-1])
                                curr_step = curr_step+1
                                print(f"{np.round(curr_step/total_steps * 100, 2)}% done")

    if len(all_grids) <= 25:
        plt.imsave(os.path.join(output_path, f"all_grids_{num_iter}_iters.png"),
                make_grid(all_grids, nrow=1).permute(1, 2, 0).numpy(),
                dpi=len(all_grids)*300)
    else:
        print(f"[INFO] Grids were NOT saved together because there were too many. {len(all_grids)} > 25")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct diffvg optimization script over a grid of options")
    parser.add_argument("--resolution", type=int, default=128, help="Resolution")
    parser.add_argument("--num_iter", type=int, default=500, help="Number of iterations for each option")
    parser.add_argument("--max_width", type=float, default=10.0, help="Max width")
    parser.add_argument("--target_image_path", type=str, default="/home/mfeuerpfeil/master/thesis/datasets/mnist_png/training/2/10024.png", help="Path to target image to optimize towards")
    parser.add_argument("--num_paths", nargs='+', type=int, default=[1], help="How many separate paths in the image")
    parser.add_argument("--segments", nargs='+', type=int, default=[2], help="Amount of segments that make a path")
    parser.add_argument("--loss_scales", nargs='+', type=float, default=[1.0], help="Scale factors for the losses")
    parser.add_argument("--stroke_widths", nargs='+', type=float, default=[1.0], help="Initial stroke widths")
    parser.add_argument("--losses", nargs='+', type=str, default=["mse", "lpips", "mix"], help="Losses to try, currently supported: mse, lpips, mix")
    parser.add_argument("--modes", nargs='+', type=str, default=["middle"], help="Modes of initialization, currently supported: middle")
    parser.add_argument("--filled", action="store_true", help="Should the path be filled (black)? Default: False")
    parser.add_argument("--primitives", nargs='+', type=str, default=["line", "cubic"], help="Primitives to optimize, currently supported: cubic, line")
    parser.add_argument("--output_dir", type=str, default="/home/mfeuerpfeil/master/thesis/images/optimization", help="Path to the directory to save the output images to")
    parser.add_argument("--verbose", action="store_true", help="Print more info like render loss etc.")
    parser.add_argument("--override", action="store_true", help="True: override already existing images, False: skip existing, default: False")


    args = parser.parse_args()
    main(args)
