import torch
import os
import vae_model
import numpy as np
from torchvision.utils import save_image
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pdb
import logging

logger1 = logging.getLogger("rnn-sample-reconstruct")
logger1.setLevel(logging.DEBUG)
fh = logging.FileHandler("logs/reconstruct.log")
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
logger1.addHandler(fh)

def train_reconstruct(x_len, x, x_recon, ix2w):
    
    logger1.info("===TRAINING (with teacher)===")

    for i in range(x_recon.size(0)):

        recon_sentence = torch.argmax(F.softmax(x_recon[i][:x_len[i]], dim=0), dim=1).tolist()
        recon_sentence = [ix2w[ix] for ix in recon_sentence]

        origin_sentence = x[i][:x_len[i]].tolist()
        origin_sentence = [ix2w[ix] for ix in origin_sentence]

        logger1.info("ORG:"+" ".join(origin_sentence))
        logger1.info("###:"+" ".join(recon_sentence)+"\n")



def sample_reconstruct(model, z_dim, ix2w):
#def sample_construct(epoch, model, q_mu_mean, q_logvar_mean, z_dim):
    logger1.info("===SAMPLE Z===")
    with torch.no_grad():
        #sample = torch.randn(64, z_dim)
        z = torch.randn(5, z_dim)
        input0 = model.embedding(torch.LongTensor([3]))
        all_decoded = model.decoder.rollout_decode(input0, z, model.embedding)
        for i, decoded in enumerate(all_decoded):
            sentence = [ix2w[ix] for ix in decoded]
            logger1.info(z[i][0:8])
            logger1.info(" ".join(sentence)+"\n")


def input_reconstruct(model, x_lengths, x_padded, ix2w):
    # encoder spits out two values here!!
    logger1.info("===GENERATE FROM TRAINING X (without teacher) ===")

    with torch.no_grad():
        embed = model.embedding(x_padded)
        q_mu, q_logvar = model.encoder(embed, x_lengths)
        z = model.sample_z_reparam(q_mu, q_logvar).squeeze()

        input0 = model.embedding(torch.LongTensor([3]))
        all_decoded = model.decoder.rollout_decode(input0, z, model.embedding)

        for j, decoded in enumerate(all_decoded):
            d_sentence = [ix2w[ix] for ix in decoded]
            e_sentence = [ix2w[ix] for ix in x_padded[j].tolist()]
            logger1.info("ORG:"+" ".join(e_sentence))
            logger1.info("### "+" ".join(d_sentence)+"\n")

