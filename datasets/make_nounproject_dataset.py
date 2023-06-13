import time
import random
from typing import List
from tqdm import tqdm
import requests
from requests_oauthlib import OAuth1
import os
from io import BytesIO
from PIL import Image
import pandas as pd

API_KEY = os.getenv("NOUN_PROJECT_API_KEY")
API_SECRET = os.getenv("NOUN_PROJECT_API_SECRET")
AUTH = OAuth1(API_KEY, API_SECRET)
QUERIES = ["hummingbird", "jellyfish", "snail", "dog", "bee", "airplane", "basketball", "beer bottle", "wall clock", "fire", "hourglass", "mailbox"]
PNG_SAVING_PATH = "/scratch1/nounproject/v1"

def do_icon_search_query(query: str, limit: int, limit_to_public_domain: int = 0, thumbnail_size: int = 200, blacklist: int = 0, include_svg: int = 0, prev_page: str = "", next_page: str = ""):
    endpoint = "https://api.thenounproject.com/v2/icon"
    query_params = {
        "query" : query,
        "limit_to_public_domain" : limit_to_public_domain,
        "thumbnail_size" : thumbnail_size,
        "blacklist" : blacklist,
        "include_svg" : include_svg,
        "limit" : limit,
        "prev_page": prev_page,
        "next_page": next_page
    }
    response = requests.get(endpoint, auth=AUTH, params=query_params)
    if(response.status_code == 200):
        return response.json()
    else:
        raise requests.exceptions.HTTPError(f"Encountered {response.status_code} error when searching for {query_params}")
    
def get_icons_from_search_response(response: dict, original_query: str) -> List[dict]:
    icons = response["icons"]
    processed_icons = []

    for i, icon in enumerate(icons):
        processed_icons.append({
        "original_query" : original_query,
        "term" : icon["term"],
        "id" : icon["id"],
        "attribution" : icon["attribution"],
        "license" : icon["license_description"],
        "tags" : ", ".join(icon["tags"]),
        "thumbnail_link" : icon["thumbnail_url"],
        'updated_at' : icon['updated_at']
        })
    return processed_icons

def crawl_icons_for_keyword(query: str, **kwargs):
    print(f"Crawling for query: {query}")
    all_icons = []
    limit = 100

    next_page = ""

    for i in tqdm(range(10)):
        response = do_icon_search_query(query, limit, next_page=next_page)
        all_icons.extend(get_icons_from_search_response(response, original_query=query))
        
        if("next_page" in response.keys()):
            next_page = response["next_page"]
        else:
            break
        
        time.sleep(random.random() * 1)

    return all_icons

if(__name__ == "__main__"):
    print(f"Worst case API usage estimate: {len(QUERIES) * 10}")

    all_icons = []
    for query in QUERIES:
        icons = crawl_icons_for_keyword(query)
        all_icons.extend(icons)

    df = pd.DataFrame(all_icons)

    print(f"Saving metadata.csv to {PNG_SAVING_PATH}")
    df.to_csv(os.path.join(PNG_SAVING_PATH, "metadata.csv"), index=False, quoting=1)

    os.makedirs(PNG_SAVING_PATH, exist_ok=True)
    print("Begin fetching thumbnails...")
    for query in df.original_query.unique():
        print(f"Fetching images for {query}")
        saving_path = os.path.join(PNG_SAVING_PATH, query)
        os.makedirs(saving_path, exist_ok=True)

        for idx, link in tqdm(enumerate(df[df["original_query"] == query]["thumbnail_link"].unique())):
            image_saving_path = os.path.join(saving_path, f"{idx}.png")
            if(os.path.exists(image_saving_path)):
                continue
            else:
                response = requests.get(link)

                if(response.status_code == 200):
                    image = Image.open(BytesIO(response.content))
                    image.save(image_saving_path)
                else:
                    raise ValueError(f"Status code for query: {query}, idx: {idx}, link: {link} was {response.status_code}")

                time.sleep(random.random()*1)