#!/usr/bin/python
# Author: Suzanna Sia

import logging
import os

def get_nn_logger(mode="train", args=None):
    
    logger = logging.getLogger("rnn-{}".format(mode))
    logger.setLevel(logging.DEBUG)

    if args.variational==1:
        fol = "vae"
    else:
        fol = "rnn"

    fh = logging.FileHandler("logs/{}/{}_z{}_h{}_bs{}_error.log".format(fol, mode, str(args.z_dim), str(args.h_dim), str(args.batch_size)))


    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    if mode=="train" or mode=="valid":
        logger.info("epoch\ttloss\tkldloss\tbceloss")

    return logger


def get_sample_logger(args):
    logger1 = logging.getLogger("rnn-sample-reconstruct")
    logger1.setLevel(logging.DEBUG)
    if args.variational==1:
        fol="vae"
    else:
        fol="rnn"
        
    fh = logging.FileHandler("logs/{}/reconstruct_z{}_h{}_bs{}.log".format(fol, str(args.z_dim),
        str(args.h_dim), str(args.batch_size)))

    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger1.addHandler(fh)

    return logger1
