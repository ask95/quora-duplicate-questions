#!/bin/bash

#source ../../miniconda2/bin/activate tensorflow
source /scratch/cluster/pgoyal/miniconda2/bin/activate tfenv
export CUDA_HOME=/stage/public/ubuntu64/local/cuda-8.0/
export LD_LIBRARY_PATH=/stage/public/ubuntu64/local/cuda-8.0/extras/CUPTI/lib64:/scratch/cluster/pgoyal/NLP/packages/cuda/cudnn/cuda/lib64
python ppdb-pred.py
