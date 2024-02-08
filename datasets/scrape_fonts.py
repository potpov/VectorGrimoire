import requests
from bs4 import BeautifulSoup
import subprocess
import zipfile
import time
import pandas as pd
from fontTools.ttLib import TTFont
import os
from glob import glob
from tqdm import tqdm
import yaml


def extract_and_delete(download_link, download_name, final_name, session):
    wget_command = f'wget ' \
                   f'--header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" --header="Cookie: {session}" ' \
                   f'-q -O {download_name} "{download_link}"'
    subprocess.run(wget_command, shell=True)

    # unzip
    with zipfile.ZipFile(download_name) as zip_ref:
        zip_ref.extractall(final_name)
    os.remove(download_name)


def unzip_and_remove(zip_dir):
    for root, dirs, files in os.walk(zip_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_file_path = os.path.join(root, file)
                extract_dir = os.path.splitext(zip_file_path)[0]
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                os.remove(zip_file_path)
                print(f"Found: '{zip_file_path}' -> '{extract_dir}'")


def convert_folder_to_ttf(main_dir):
    extensions = ['otf', 'eot', 'woff', 'woff2']
    counter = 0
    for extension in extensions:
        font_files = glob(os.path.join(main_dir, f'**/*.{extension}'), recursive=True)
        for font_file in font_files:
            super_dir = os.path.dirname(font_file)
            output_path = os.path.join(super_dir, f"{os.path.splitext(os.path.basename(font_file))[0]}.ttf")
            # Check if the TTF file already exists, and skip if it does
            if not os.path.exists(output_path):
                try:
                    font = TTFont(font_file)
                    font.save(output_path)
                    counter += 1
                except Exception as e:
                    print(f"Error converting {font_file}: {e}")
            else:
                print(f"TTF file {output_path} already exists. Skipping.")
    print(f"{counter} fonts converted into ttf")


def scrape_allfreefonts(params):
    # even the homepage because we are greedy!
    pages = ["https://www.allfreefonts.co/"] + [f"https://www.allfreefonts.co/page/{i}/" for i in range(1, 815)]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Cookie': f'aiADB={params["aiadb_cookie"]}'
    }
    counter = 0
    skipped = []
    metadata = []
    for page in tqdm(pages, total=len(pages)):
        response = requests.get(page, headers=headers)
        html_content = response.content

        soup = BeautifulSoup(html_content, 'html.parser')
        articles = soup.find_all('article')
        for article in articles:
            # Extract font name from the article title
            article_title = article.find('h2', class_='entry-title').text.strip()
            font_name = article_title.replace("Font", "").strip().replace(" ", "-").lower()

            # Extracting the download link from the download page
            font_download_page = f'https://www.allfreefonts.co/download/{font_name}/'
            response = requests.get(font_download_page, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            download_link = soup.find('a', href=lambda x: x and '.zip' in x)
            # Extracting categories
            font_page_link = article.find("a", class_="entry-title-link")["href"]
            article_soup = BeautifulSoup(requests.get(font_page_link, headers=headers).content, 'html.parser')
            breadcrumb_div = article_soup.find('div', class_='breadcrumb')

            # removing last (font name) and first one ("Home")
            try:
                tags = [a.lower().strip() for a in breadcrumb_div.text.split("›")[1:-1]]
            except Exception as e:
                tags = []

            if not os.path.exists(os.path.join(params['fonts_dir'], f"{counter}_{font_name}")):
                try:
                    download_link = download_link['href']
                    extract_and_delete(
                        download_link=download_link,
                        download_name=os.path.join(params['fonts_dir'], f"{counter}_{font_name}.zip"),
                        final_name=os.path.join(params['fonts_dir'], f"{counter}_{font_name}"),
                        session=f'aiADB={params["aiadb_cookie"]}'
                    )
                    time.sleep(1)  # lil delay to not piss the firewall off
                except Exception as e:
                    skipped.append(font_name)
                    continue
            else:
                print(os.path.join(params['fonts_dir'], f"{counter}_{font_name}") + " already exists")

            counter += 1
            metadata.append({
                "filename": f"{counter}_{font_name}",
                "tags": tags
            })

    print("Skipped: ", {skipped})
    print("Saving metadata...")
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(params['fonts_dir'], "metadata.csv"), index=False)
    print("Thats' all folks!")


def scrape_dafont(params):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Cookie': f'PHPSESSID={params["session_cookie"]}'
    }
    counter = 0
    metadata = []
    skipped = []


    pages = ["https://www.dafont.com/new.php?nup=3"] + [f"https://www.dafont.com/new.php?page={i}&nup=3" for i in range(1, 409)]
    for page in tqdm(pages, total=len(pages)):
        response = requests.get(page, headers=headers)
        html_content = response.content

        soup = BeautifulSoup(html_content, 'html.parser')
        divs = soup.find_all('div', class_='lv1right dfbg')
        for div in divs:
            preview_div = div.find_next('div', class_='preview')
            font_name = preview_div.find('a')['href'].split('=')[-1].replace(".font", "")
            download_link = "https:" + div.find_next('div', class_='dlbox').find('a', class_='dl')['href']
            tags = [a.text for a in div.find_all('a')]

            try:
                extract_and_delete(
                    download_link=download_link,
                    download_name=os.path.join(params['fonts_dir'], f"{counter}_{font_name}.zip"),
                    final_name=os.path.join(params['fonts_dir'], f"{counter}_{font_name}"),
                    session=f'PHPSESSID={params["session_cookie"]}'
                )
            except Exception as e:
                skipped.append(font_name)
                continue

            metadata.append({
                "filename": "{counter}_{font_name}",
                "tags": tags
            })
            counter += 1

    print("Skipped: ", {skipped})
    print("Saving metadata...")
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(params['fonts_dir'], "metadata.csv"), index=False)
    print("Thats' all folks!")


if __name__ == '__main__':
    with open("font_paths.yaml", "r") as stream:
        config = yaml.safe_load(stream)["fonts"]

    print("scraping www.allfreefonts.co *evil smile*")
    params = config["allfonts"]
    scrape_allfreefonts(params)
    unzip_and_remove(params['fonts_dir'])  # some files in allfreefonts have zip with bonus fonts inside
    convert_folder_to_ttf(params['fonts_dir'])

    print("scraping dafont.com *evil smile*")
    params = config["dafont"]
    scrape_dafont(params)
    convert_folder_to_ttf(params['fonts_dir'])
