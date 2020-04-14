#!/usr/bin/env bash
#SBATCH -A C3SE2020-1-18 -p vera
#SBATCH --gres=gpu:V100:1
#SBATCH -t 0-2:10:00
#SBATCH --mail-user=gablinde@student.chalmers.se
#SBATCH --mail-type=ALL
python Distributed_mp.py

