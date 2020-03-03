#!/usr/bin/env bash
#SBATCH -A C3SE2020-1-18 -p vera
#SBATCH --gres=gpu:T4:1     # allocates 1 T4 GPU (and a full node)
#SBATCH -t 0-00:06:00
python learner_alone.py

