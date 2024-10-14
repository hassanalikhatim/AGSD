#!/usr/bin/env bash

cd $HOME/
git clone "https://github.com/hassanalikhatim/download_agsd_pretrained_models.git"
cd download_pretrained_models
git remote -v
git pull origin main
git remote remove origin
mkdir ../__all_results__/_p1_agsd/
# mkdir ../__all_results__/_p1_agsd/results_1/
mv results_agsd_some/ ../__all_results__/_p1_agsd/results_1/
cd ..
rm -rf download_pretrained_models
ls
cd $HOME/code/AGSD/
