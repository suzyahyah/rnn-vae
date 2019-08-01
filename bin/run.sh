#!/usr/bin/env bash

source activate rnn-vae

if [ "$HOSTNAME" == "DESKTOP-KFANQCM" ]; then
  CUDA=-1
  FN=data/ptb.train.txt
else
  CUDA=`free-gpu`
  FN=data/ptb.train.txt
fi


declare -A M
declare -A F

F=(
["OLD_EMB"]=data/embeddings/glove_w2vec.txt
["NEW_EMB"]=data/embeddings/glove_matrix
["VOCAB_FN"]=data/embeddings/glove.vocab
["TRAIN_FN"]=data/ptb.train.txt
["VALID_FN"]=data/ptb.valid.txt
)

M=(
["CUDA"]=$CUDA
#["CUDA"]=-1
#["TRAIN_FN_X"]=data/all/train_x.txt
#["TRAIN_FN_Y"]=data/all/train_y.txt
["NUM_EPOCHS"]=500
["NWORDS"]=10000
["L_EPOCH"]=0
["RNNGATE"]=$2
["FRAMEWORK"]=$1
["EMBEDDING_DIM"]=350 # used in the paper
["HIDDEN_DIM"]=200 # 
["LATENT_DIM"]=12
["BATCH_SIZE"]=128
["N_LAYERS"]=1
["DELTA_WEIGHT"]=False
["UNIVERSAL_EMBED"]=True
["WORD_DROPOUT"]=$3
["KL_ANNEAL_STEPS"]=5000
# KL Annealing for the first 5000 training steps
)

PY="python code/main.py"

for var in "${!M[@]}"
do
  PY+=" --${var,,} ${M[$var]}"
  echo "| $var:${M[$var]}"
done

for var in "${!F[@]}"
do
  PY+=" --${var,,} ${F[$var]}"
  echo "| $var:${F[$var]}"
done


eval $PY
