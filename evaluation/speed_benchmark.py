import os
import time
from models.clip_draw import Predictor
import yaml
from eval_ART_strokes import load_model_from_basepath, load_stage2_model_from_basepath
import torch
from tqdm import tqdm
import math

if __name__ == '__main__':
    # print("Speed benchmarking")
    # print("#"*50)
    # print("EVALUATING CLIPDRAW")
    # print("#"*50)
    #
    # p = Predictor()
    # p.setup()
    # time_results = []
    # # "The letter A in italic font"
    # # The icon of a light bulb in black and white.
    # for i in range(5):
    #     start_time = time.time()
    #     output = p.predict(prompt="The icon of a user in black and white.", filename=f"output_{i}.png")
    #     elapsed_time = time.time() - start_time
    #     time_results.append(elapsed_time)
    #     print(f"Elapsed time: {elapsed_time:.2f}s")
    # print(f"Average time: {sum(time_results)/len(time_results):.2f}s")


    print("#"*50)
    print("EVALUATING SVG2PDF")
    print("#"*50)

    # CONFIG EVERYTHING HERE
    STAGE2_BASE_PATH = "/raid/marco.cipriano/results/svg/Grimoire/ART/figr8/nseg=4_ncode=2_lseg=5_alpha01_256grid"
    NUM_CONTEXT_STROKES = [0]

    # ---------------------
    BATCH_SIZE = 1
    RENDER_WIDTH = 480
    GLOBAL_STROKE_WIDTH = 0.7
    TEMPERATURE = 0.1
    SAMPLING_METHOD = None
    MAX_DIST_FRAC = 4 / 72
    FIXING_METHODS = []
    NUM_SAMPLES = 4
    # ---------------------

    BASE_OUT_DIR = os.path.join(STAGE2_BASE_PATH, "test")
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    device = torch.device("cuda")
    # load config to extract stage1 params
    config = yaml.load(open(os.path.join(STAGE2_BASE_PATH, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    # config = map_wand_config(config)

    # load VSQ
    vsq_base_path = config["stage1_params"]["checkpoint_path"].split("checkpoints")[0]
    vsq_model = load_model_from_basepath(vsq_base_path, device=device)[0]

    # get model and data module
    stage_2_model, stage2_dm, stage2_config = load_stage2_model_from_basepath(vsq_model, STAGE2_BASE_PATH, device=device)
    stage2_dm.test_batch_size = BATCH_SIZE

    # generation pipeline
    dl = stage2_dm.test_dataloader()

    vq_context = 0
    curr_svg_out_dir = os.path.join("/tmp/grim_speed_test")
    os.makedirs(curr_svg_out_dir, exist_ok=True)

    generations = []
    captions = []
    all_ids = []
    time_results = []
    print(f"Generating...")
    for counter, (text_tokens, attention_mask, vq_tokens, _, svg_ids) in tqdm(enumerate(dl)):
        start_time = time.time()
        text_tokens = text_tokens.to(device)
        attention_mask = attention_mask.to(device)
        curr_vq_tokens = vq_tokens[:, :1].clone().to(device)
        generation, reason = stage_2_model.generate(text_tokens, attention_mask, curr_vq_tokens, temperature=TEMPERATURE, sampling_method=SAMPLING_METHOD)
        drawing = stage_2_model.tokenizer._tokens_to_svg_drawing(
            generation,
            global_stroke_width=GLOBAL_STROKE_WIDTH,
            post_process=False,
            num_strokes_to_paint=0
        )
        drawing.saveas(os.path.join(curr_svg_out_dir, f"test.svg"))
        elapsed_time = time.time() - start_time
        time_results.append(elapsed_time)
        if counter > NUM_SAMPLES:
            break

    print(f"Average time: {sum(time_results)/len(time_results):.2f}s")