import numpy as np
from tqdm import tqdm
import random
import json
from thesis.dataset import GlyphazznStage1Dataset

NUM_SAMPLES = 50000

def get_fraction_dict(all_num_shapes: np.ndarray, indices: list):
    return_dict = {
        "len": len(all_num_shapes),
        "indices": indices,
        "frac_54": (all_num_shapes <= 54).sum() / len(all_num_shapes),
        "frac_64": (all_num_shapes <= 64).sum() / len(all_num_shapes),
        "frac_118": (all_num_shapes <= 118).sum() / len(all_num_shapes),
        "frac_128": (all_num_shapes <= 128).sum() / len(all_num_shapes),
        "frac_246": (all_num_shapes <= 246).sum() / len(all_num_shapes),
        "frac_256": (all_num_shapes <= 256).sum() / len(all_num_shapes),
        "frac_502": (all_num_shapes <= 502).sum() / len(all_num_shapes),
        "frac_512": (all_num_shapes <= 512).sum() / len(all_num_shapes),
    }
    print(return_dict)
    return return_dict

min_len = 1.0

ds_7 = GlyphazznStage1Dataset(
    csv_path="/scratch2/moritz_data/figr8/stage1.csv",
    channels=3,
    width=128,
    train=True,
    subset="all",
    individual_min_length=min_len,
    individual_max_length=7.0,
    max_shapes_per_svg=1000,
)

ds_5 = GlyphazznStage1Dataset(
    csv_path="/scratch2/moritz_data/figr8/stage1.csv",
    channels=3,
    width=128,
    train=True,
    subset="all",
    individual_min_length=min_len,
    individual_max_length=5.0,
    max_shapes_per_svg=1000,
)

ds_3 = GlyphazznStage1Dataset(
    csv_path="/scratch2/moritz_data/figr8/stage1.csv",
    channels=3,
    width=128,
    train=True,
    subset="all",
    individual_min_length=min_len,
    individual_max_length=3.0,
    max_shapes_per_svg=1000,
)




all_dicts = []
all_num_shapes_7 = []
all_num_shapes_5 = []
all_num_shapes_3 = []

random_idxs = random.sample(range(len(ds_7)), NUM_SAMPLES)

for i in tqdm(random_idxs, total=len(random_idxs)):
    all_num_shapes_7.append(ds_7[i][0].shape[0])
all_dicts.append(get_fraction_dict(np.array(all_num_shapes_7), random_idxs))
with open("/scratch2/moritz_data/figr8/stats/fraction_dicts.json", "w") as f:
    json.dump(all_dicts, f)
np.save(
    "/scratch2/moritz_data/figr8/stats/all_num_shapes_7.npy", np.array(all_num_shapes_7)
)

for i in tqdm(random_idxs, total=len(random_idxs)):
    all_num_shapes_5.append(ds_5[i][0].shape[0])
all_dicts.append(get_fraction_dict(np.array(all_num_shapes_5), random_idxs))
with open("/scratch2/moritz_data/figr8/stats/fraction_dicts.json", "w") as f:
    json.dump(all_dicts, f)
np.save(
    "/scratch2/moritz_data/figr8/stats/all_num_shapes_5.npy", np.array(all_num_shapes_5)
)

for i in tqdm(random_idxs, total=len(random_idxs)):
    all_num_shapes_3.append(ds_3[i][0].shape[0])
all_dicts.append(get_fraction_dict(np.array(all_num_shapes_3), random_idxs))
with open("/scratch2/moritz_data/figr8/stats/fraction_dicts.json", "w") as f:
    json.dump(all_dicts, f)
np.save(
    "/scratch2/moritz_data/figr8/stats/all_num_shapes_3.npy", np.array(all_num_shapes_3)
)

print("DONE")
