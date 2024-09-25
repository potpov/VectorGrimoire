# paper notes

## VSQ MODULE CONFIGURATIONS

save directory: `/raid/marco.cipriano/results/svg/Grimoire/VSQ`

### experiments

| Name                                | Notes                     |
|-------------------------------------|---------------------------|
| VSQ_MNIST_BW_P128_T6_P20_TH0.1      |                           |
| VSQ_MNIST_BW_P128_T14_P20_TH0.2     | larger sequence for ART   |
| VSQ_MNIST_COLOR_P128_T3_P20         |                           |
| VSQ_MNIST_BW_P128_T14_P20_TH0.2_S64 | more segmentens per shape |

final experiments path



| Name           | Notes                                                                                         |
|----------------|-----------------------------------------------------------------------------------------------|
| FIGR8 - ART    | /raid/marco.cipriano/results/svg/Grimoire/ART/figr8/nseg=4_ncode=2_lseg=5_alpha01_256grid     |
| FIGR8 - VSQ    | /raid/marco.cipriano/results/svg/Grimoire/VSQ/figr8/VSQ_nseg=4_ncode=2_lseg=5.0_alpha=0.1None |
| FONTS - ART    |                                                                                               |
| FONTS - VSQ    | /raid/marco.cipriano/results/svg/Grimoire/VSQ/VSQ_FONTS                                       |
| MNIST - ART    | /raid/marco.cipriano/results/svg/Grimoire/ART/ART_MNIST_BW_P6T0.2                             |
| MNIST - VSQ    | /raid/marco.cipriano/results/svg/Grimoire/VSQ/VSQ_MNIST_BW_P128_T6_P20_TH0.1                  |
| MNIST - im2vec |                                                                                               |


## Datasets

### MNIST
```
/raid/marco.cipriano/data/SVG/Grimoire/MNIST/
├─ mnist_png/
│  ├─ testing/
│  ├─ training/
├─ mnist_tokenized/
│  ├─ P128_T14_P20_TH0.2
├─ mnist_pretiled/
│  ├─ P128_T14_P20_TH0.2
│  ├─ ...
```

# Install
Master's thesis on conditional SVG generative models.
## Setup
First of all, make a new python env and install requirements.txt:
```bash
$ conda create --name SVG python=3.10
$ conda activate SVG
```
Clone and Install [diffsvg](https://github.com/BachiLi/diffvg) as explained in the repo:
```bash
git clone git@github.com:BachiLi/diffvg.git
git submodule update --init --recursive
conda install -y pytorch torchvision -c pytorch
conda install -y numpy
conda install -y scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
python setup.py install
```
Install our requirements:
```bash
$ conda env update -n SVG --file requirements.yaml
```

## process font datasets
NOTE: all scripts must be modified, paths are hard-coded


- fonts are initally in .ttf format
- convert them to svgs in `thesis/datasets/ttf_to_svg.py`

### process for causal auto-regressive prediction
- then we need to normalize and split them, which is done in the deepsvg, namely with `deepsvg/dataset/preprocess.py`
- after that we need to generate the rasterized time-split version with `thesis/datasets/make_causal_positional_dataset.py`

### process for VQ-VAE pipeline
For the first step we would actually just need SVGs, not even normalized as the dataloader `CenterShapeLayersFromSVGDataset` automatically splits paths. However, for the second stage (transformer training) we need normalized versions of the SVG data anyway (for the positions), so just generate the normalized versions first and you're good.

- normalize with deepsvg in `deepsvg/dataset/preprocess.py` (must be run with `python -m dataset.preprocess` in the deepsvg directory)

## DATA
### Fonts
#### SVG-VAE - Glyphazzn
This dataset is often referenced in the following related work as the "SVG-Fonts" dataset. It contains 14M examples across 62 characters (a-z, A-Z, 0-9). That is on average 225k examples for each glyph. At the time of this writing (11/2023) downloading the fonts from this dataset results in many 404 errors and no responses. The download of the remaining fonts can be found in `thesis/datasets/download_glyphazzn_dataset.py`.

After downloading only the ttf files, we get:
- 28136 fonts

Note: on the conversion to svg, some fonts failed for only a subset of glyphs. E.g. many fonts just could not generate numbers, but all letters without issues.

#### Google Fonts
[GoogleFonts](https://github.com/google/fonts)

#### Deepvecfont - subset of Glyphazzn
This work from 2021 has a download available for a subset from SVG-VAE. Can be found [here](https://github.com/yizhiwang96/deepvecfont#customize-your-own-dataset).
- 27383 train fonts, only ~14500 .ttf files
- 3045 test fonts

They are all in .ttf, .otf, or .pfb format. I have continued with downloading the still available fonts in .ttf format myself as this yielded more fonts in total and the conversion script was already in place.


# DEPRECATION ALERT
### MNIST
To download the dataset, please run the following scripts:
```bash
$ PATH_TO_REPO=YOUR_PATH/thesis bash ./datasets/download.sh
```

### MNIST++
After downloading MNIST, run `datasets/setup.py`:
```bash
$ python datasets/setup.py
```

### Icons8 - COMING SOON
It is currently not clear, how we want to use this dataset. It is not suited for filled shapes, and using only outlines results in overly complex compositions as the original datasource was colored and shaded.

TODO:
- [ ] agree on a format and make the dataset
### TheNounProject
TheNounProject is a custom scraped dataset. It was generated with the following procedure:
1. Register an account at [The Noun Project](https://thenounproject.com/)
2. Get your API keys at the [developer portal](https://thenounproject.com/developers/apps)
3. Write your API key and secret inside an .env file in the root directory as
```bash
NOUN_PROJECT_API_KEY="key"
NOUN_PROJECT_API_SECRET="private"
```
4. Modify the global variables in `datasets/make_nounproject_dataset.py` accordingly

Then, run the script:
```bash
$ python datasets/make_nounproject_dataset.py
```

TODO:
- [ ] Provide a download link for reproducibility

### Emoji
We're using the open-source Twitter emojis. To reproduce:
1. Clone the [twemoji repository](https://github.com/twitter/twemoji)
2. Modify global variables in `datasets/make_emoji_dataset.py`

Then, run:
```bash
$ python datasets/make_emoji_dataset.py
```
