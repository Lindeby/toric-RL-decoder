#!/usr/bin/env bash
#SBATCH -A C3SE2020-1-18 -p vera
#SBATCH --gres=gpu:V100:1   # allocates 1 V100 GPU (and half the node)
#SBATCH -t 0-00:10:00
python learner_alone.py

