# thesis
Master's thesis on conditional SVG generative models.
## Setup
First of all, make a new python env and install requirements.txt:
```bash
$ conda create --name thesis python=3.10
$ conda activate thesis
$ pip install -r requirements.txt
```

To download the datasets, please run the following scripts:
```bash
$ PATH_TO_REPO=YOUR_PATH/thesis bash ./datasets/download.sh
```

After that, run the `datasets/setup.py`:
```bash
$ python datasets/setup.py
```
