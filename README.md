# thesis
Master's thesis on conditional SVG generative models.
## Setup
First of all, make a new python env and install requirements.txt:
```bash
$ conda create --name thesis python=3.10
$ conda activate thesis
$ pip install -r requirements.txt
```
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
3. Register your API key and secret as environment variables
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
