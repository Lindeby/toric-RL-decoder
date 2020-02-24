#!/usr/bin/env bash
#SBATCH -A C3SE2020-1-18 -p vera
#SBATCH -n 32
#SBATCH -t 0-00:10:00

python actor_alone.py

