#!/bin/bash
#PBS -l select=1:ncpus=8:mpiprocs=4

export CUDA_HOME="/vol/cuda/9.0.176"
export CUDA_HOME2="/vol/cuda/8.0.61-cudnn.7.0.2"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_ROOT=${CUDA_HOME}/bin:${CUDA_HOME2}/bin
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH":"${CUDA_HOME2}/lib64:$LD_LIBRARY_PATH"
export CPATH="£{CUDA_HOME}/include:$CPATH"

export PATH="/vol/gpudata/wt814/miniconda3/envs/venv1/bin:$PATH"

echo Today is $( date )
echo This is $( /bin/hostname )
# cd /vol/gpudata/wt814/
#
# cd ~/PROJECT
# echo The current working directory is $( pwd )
# source ~/.bashrc
# source activate tensorflow_env
# cd tensorflow_cifar_10
# cd 651-1000
# python new_keras_651_1000.py > out_out.log
# uptime


cd /vol/gpudata/nj2217/TEST
echo The current working directory is $( pwd )
source ~/.bashrc
source activate tensorflow_env
python mnist_test.py
uptime
