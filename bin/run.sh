#!/usr/bin/env bash

source activate rnn-vae

declare -A M

M=(
#["CUDA"]=`free-gpu`
["NUM_EPOCHS"]=500
["NWORDS"]=20000
["L_EPOCH"]=0
["RNNGATE"]=gru
["FRAMEWORK"]=$1
["HIDDEN_DIM"]=512
["LATENT_DIM"]=$2
["BATCH_SIZE"]=$3
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
