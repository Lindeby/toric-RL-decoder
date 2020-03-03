#!/usr/bin/sh 

#mkdir $SNIC_NOBACKUP/my_python

module load GCC/8.2.0-2.31.1  CUDA/10.1.105  OpenMPI/3.1.3  

module load PyTorch/1.2.0-Python-3.7.2

#In from inside gym_toric_code run:
#pip install -e ./ --prefix $SNIC_NOBACKUP/my_python 

export PYTHONPATH=$PYTHONPATH:$SNIC_NOBACKUP/my_python/lib/python3.7/site-packages/



