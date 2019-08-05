#!/usr/bin/env bash
# ztriple
source activate rnn-vae

if [ "$HOSTNAME" == "DESKTOP-KFANQCM" ]; then
  CUDA=-1
else
  CUDA=`free-gpu`
fi

declare -A M
declare -A F
declare -A Z

F=(
["OLD_EMB"]=data/embeddings/all/glove_w2vec.txt
["NEW_EMB"]=data/embeddings/all/glove_matrix
["VOCAB_FN"]=data/embeddings/all/glove.vocab
# just use the bigger one, no harm
["TRAIN_FN"]=data/ptb.train.txt
["VALID_FN"]=data/ptb.valid.txt
)

M=(
["CUDA"]=$CUDA
["NUM_EPOCHS"]=500
["NWORDS"]=10000
["L_EPOCH"]=0
["RNNGATE"]=lstm
["FRAMEWORK"]=vae
["HIDDEN_DIM"]=200
["LATENT_DIM"]=12
["EMBEDDING_DIM"]=300
["BATCH_SIZE"]=128
["N_LAYERS"]=1
["DELTA_WEIGHT"]=False
)

Z=(
["UNIVERSAL_EMBED"]=False
["SCALE_PZVAR"]=$1
["KL_ANNEAL_STEPS"]=$2
["WORD_DROPOUT"]=$3
["BOW"]=$4
)

echo "CUDA device:", ${M[CUDA]}
export CUDA_VISIBILE_DEVICES=${M[CUDA]}

PY="python code/main.py"

echo "===Model Params==="
for var in "${!M[@]}"
do
  PY+=" --${var,,} ${M[$var]}"
  echo "| $var:${M[$var]}"
done

echo "===File Names==="
for var in "${!F[@]}"
do
  PY+=" --${var,,} ${F[$var]}"
  echo "| $var:${F[$var]}"
done

echo "===ELBO tweaks==="
for var in "${!Z[@]}"
do
  PY+=" --${var,,} ${Z[$var]}"
  echo "| $var:${Z[$var]}"
done

eval $PY
