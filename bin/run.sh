#!/usr/bin/env bash

source activate rnn-vae

declare -A M

M=(
["CUDA"]=`free-gpu`
["NUM_EPOCHS"]=500
["BATCH_SIZE"]=32
["NWORDS"]=10000
["L_EPOCH"]=0
["VARIATIONAL"]=1
)
echo "CUDA device:", ${M[CUDA]}
export CUDA_VISIBILE_DEVICES=${M[CUDA]}

PY="python code/main.py"

for var in "${!M[@]}"
do
  PY+=" --${var,,} ${M[$var]}"
  echo "| $var:${M[$var]}"
done

eval $PY
