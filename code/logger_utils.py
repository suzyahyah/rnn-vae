#!/usr/bin/python
# Author: Suzanna Sia

import logging

def get_nn_logger(mode="train", variational=1):
    
    logger = logging.getLogger("rnn-{}".format(mode))
    logger.setLevel(logging.DEBUG)

    if variational==1:
        fh = logging.FileHandler("logs/vae/{}_error.log".format(mode))
    else:
        fh = logging.FileHandler("logs/rnn/{}_error.log".format(mode))

    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.info("epoch\ttloss\tkldloss\tbceloss")

    return logger



