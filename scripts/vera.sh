#!/usr/bin/env bash
#SBATCH -A C3SE2020-1-18 -p vera
#SBATCH -n 16
#SBATCH -t 0-00:10:00

python actor_alone.py 1 5000

