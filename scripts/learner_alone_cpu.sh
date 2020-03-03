#!/usr/bin/env bash
#SBATCH -A C3SE2020-1-18 -p vera
#SBATCH -n 1
#SBATCH -t 0-00:10:00

python learner_alone.py

