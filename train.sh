#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
LOG="log/train_${1}.txt.$(date +'%Y-%m-%d_%H-%M-%S')"
exec &> >(tee -a "${LOG}")

source /n/fs/pvl-ins-seg/anaconda3/bin/activate CornerNet_Lite
./train.py ${1} --iter ${2} --workers ${3}
