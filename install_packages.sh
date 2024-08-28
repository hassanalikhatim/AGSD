#!/usr/bin/env bash

conda create -n agsd_artifact python=3.12
conda activate agsd_artifact
pip install -r requirements.txt