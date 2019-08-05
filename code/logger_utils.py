#!/usr/bin/python
# Author: Suzanna Sia

import logging
import os
import pdb


def log_file(args):
    fol = "{}-{}".format(args.framework, args.rnngate)


    fil = "z{}-h{}/bs{}-emb{}/s{}-wd{}-ksteps{}-bow{}".format(str(args.z_dim), 
                                                str(args.h_dim),
                                                str(args.batch_size),
                                                args.universal_embed,
                                                str(args.scale_pzvar),
                                                str(args.word_dropout),
                                                str(args.kl_anneal_steps),
                                                args.bow) 
                                            
    return os.path.join(fol, fil)

def get_nn_logger(mode="train", args=None):
    
    logger = logging.getLogger("rnn-{}".format(mode))
    logger.setLevel(logging.DEBUG)

    fil = log_file(args)
    fil = 'logs/ptb/{}/{}.err'.format(mode, fil)

    fol = os.path.dirname(fil)

    if not os.path.isdir(fol):
        os.makedirs(fol)
  

    fh = logging.FileHandler(fil)
    formatter = logging.Formatter('%(message)s')

    fh.setFormatter(formatter)

    logger.addHandler(fh)
    if mode=="train" or mode=="valid":
        logger.info("epoch\ttloss\tkldloss\tbceloss")

    return logger


def get_sample_logger(mode="train", args=None):
    logger1 = logging.getLogger("rnn-sample-reconstruct")
    logger1.setLevel(logging.DEBUG)

    fil = log_file(args)
    fil = 'logs/ptb/{}/{}.log'.format(mode, fil)

    fol = os.path.dirname(fil)
    if not os.path.isdir(fol):
        os.makedirs(fol)
 
    fh = logging.FileHandler(fil)
        
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger1.addHandler(fh)

    return logger1

def get_save_dir(args):

    fil = log_file(args)
    fil = 'models/ptb/{}/models'.format(fil)

    fol = os.path.dirname(fil)
    if not os.path.isdir(fol):
        os.makedirs(fol)
 
    return fil
