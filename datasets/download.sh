#!/bin/bash

# MNIST dataset
wget -O /tmp/mnist_png.tar.gz https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
tar xzf /tmp/mnist_png.tar.gz -C $PATH_TO_REPO/datasets/
