

import requests
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import os
import sys
# download txt file from https://storage.googleapis.com/magentadata/models/svg_vae/glyphazzn_urls.txt

failed_counter = 0
df = pd.read_csv('glyphazzn_urls.txt', names=["id", "split", "url"])
df_ttf = df[df["url"].str.lower().str.contains(".ttf")]

def download_font(url, save_path, failed_counter):
    try:
        response = requests.get(url)
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            sys.exit()
        failed_counter =failed_counter+ 1
        print(f"Failed to get response.")
        return
    
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Font downloaded successfully to {save_path}")
    else:
        failed_counter =failed_counter+ 1
        print(f"Failed to download font. Status code: {response.status_code}, url: {url}")
    return failed_counter

for i, row in df_ttf.iterrows():
    if i % 100 == 0:
        print(f"currently at {100*i / df_ttf.shape[0]:.2f}%")
        print(f"Fail counter: {failed_counter}")
    font_url = row["url"]
    curr_id = str(row["id"])
    base_path = "/scratch2/moritz_data/glyphazzn/font_files"
    save_location = os.path.join(base_path, curr_id+".ttf")
    if os.path.exists(save_location):
        print(f"{save_location} exists already.")
        continue
    else:
        failed_counter = download_font(font_url, save_location, failed_counter)
        # print(save_location, font_url)
    time.sleep(0.002 * np.random.randint(1, 3))
