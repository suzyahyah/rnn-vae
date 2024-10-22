import torch
import os
import vae_model
import numpy as np
from torch.nn import functional as F
import pdb
import logging

def train_reconstruct(logger1, x_len, x, x_recon, ix2w):
    
    logger1.info("===TRAINING (with teacher)===")

    for i in range(x_recon.size(0)):

        recon_sentence = torch.argmax(F.softmax(x_recon[i][:x_len[i]], dim=1), dim=1).tolist()
        recon_sentence = [ix2w[ix] for ix in recon_sentence]

        origin_sentence = x[i][:x_len[i]].tolist()
        origin_sentence = [ix2w[ix] for ix in origin_sentence]
        
        logger1.info("ORG:"+" ".join(origin_sentence))
        logger1.info("###:"+" ".join(recon_sentence)+"\n")



def sample_reconstruct(logger1, epoch, model, input0, z, ix2w):
#def sample_construct(epoch, model, q_mu_mean, q_logvar_mean, z_dim):
    logger1.info(">>>EPOCH:"+str(epoch))
    logger1.info("===SAMPLE Z===")
    with torch.no_grad():
        #sample = torch.randn(64, z_dim)
        all_decoded, _ = model.decoder.rollout_decode(input0, z, model.embedding, 25)
        for i, decoded in enumerate(all_decoded):
            sentence = [ix2w[ix] for ix in decoded]
            logger1.info(z[i][0:8])
            logger1.info(" ".join(sentence)+"\n")
#            if i==0:
#                print(" ".join(sentence))


def input_reconstruct(logger1, model, x_lengths, x_padded, input0, ix2w, device):
    # encoder spits out two values here!!
    logger1.info("===GENERATE FROM TRAINING X (without teacher) ===")

    with torch.no_grad():
        embed = model.embedding(x_padded)
        q_mu, q_logvar = model.encoder(embed, x_lengths)

        if model.framework=="vae":
            z = model.sample_z_reparam(q_mu, q_logvar).squeeze()
        else:
            z = q_mu.squeeze()

        all_decoded,_ = model.decoder.rollout_decode(input0, z, model.embedding, 25)

        for j, decoded in enumerate(all_decoded):
            d_sentence = [ix2w[ix] for ix in decoded]
            e_sentence = [ix2w[ix] for ix in x_padded[j].tolist()]
            logger1.info("ORG:"+" ".join(e_sentence))
            logger1.info("### "+" ".join(d_sentence)+"\n")

