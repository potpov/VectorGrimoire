conda activate thesis
cd /home/mfeuerpfeil/master/thesis
CUDA_VISIBLE_DEVICES=0 python direct_optimization.py --resolution 128 --num_iter 1000 --max_width 10.0 --target_image_path datasets/mnist_png/training/2/10024.png --segments 2 3 4 --loss_scales 1.0 --stroke_widths 1.0 --losses mse lpips mix --modes middle --output_dir images/optimization
