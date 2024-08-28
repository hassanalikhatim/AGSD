#!/usr/bin/env bash

echo "Downloading miniconda to setup environments"
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda"
source "$HOME/miniconda/bin/activate"
conda init
conda list
conda info --envs
conda update -n base -c defaults conda
