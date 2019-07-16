#!/usr/bin/python
# Author: Suzanna Sia

import logging

def get_nn_logger(mode="train"):
    
    logger = logging.getLogger("rnn-vae-{}".format(mode))
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("logs/{}_error.log".format(mode))
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.info("epoch\ttloss\tkldloss\tbceloss")

    return logger



