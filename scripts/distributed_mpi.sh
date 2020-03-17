#!/usr/bin/env bash
#SBATCH -A C3SE2020-1-18 -p vera
#SBATCH --gres=gpu:T4:1     # allocates 1 T4 GPU (and a full node)
#SBATCH -t 0-01:00:00
mpiexec -n 32 python start_mpi.py

