#!/usr/bin/env bash

conda create -n agsd_artifact python=3.12
conda activate agsd_artifact
conda env update --name agsd_artifact --file requirements_agsd.yml
conda deactivate
conda activate agsd_artifact
