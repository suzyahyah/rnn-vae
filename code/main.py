import utils
import vae_model
import numpy as np
import torch
from torch import optim
import os
import pdb
import vis
import logging

np.random.seed(0)
num_epochs=50
SAVE_PATH="models/rnn-vae"
LOG_PATH="logs/errors"
batch_size=25
nwords=10000

logger = logging.getLogger("rnn-vae")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("logs/errors.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


if __name__=="__main__":

    train_dataset = utils.TextDataset(fn="data/ptb.test.txt", nwords=nwords)
    train_dataset.make_ix_dicts(train_dataset.all_words)
    train_dataloader = utils.get_dataloader(train_dataset, batch_size=batch_size)

    valid_dataset = utils.TextDataset(fn="data/ptb.valid.txt", nwords=nwords)
    valid_dataset.make_ix_dicts(valid_dataset.all_words)
    valid_dataloader = utils.get_dataloader(valid_dataset, batch_size=batch_size)

    model = vae_model.RNNVAE(nwords=train_dataset.vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(num_epochs):
        print("epoch:", epoch)
        running_loss = 0.0
        r_bceloss = 0.0
        r_kldloss = 0.0

        model.train()
        for i, (x_len, x, y, eos_seq) in enumerate(train_dataloader):
            # x_recon, x doesnt start with eos
            # y ends with eos 
            x_recon, q_mu, q_logvar = model(x_len, x, eos_seq)
            bce, kld = model.loss_fn(y, x_recon, q_mu, q_logvar, x_len)

            loss = bce/batch_size + kld/batch_size
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            r_kldloss += (kld/batch_size).item()
            r_bceloss += (bce/batch_size).item()
            
            if i%100==0:
                vis.sample_reconstruct(model, 20, train_dataset.ix2w)
                vis.input_reconstruct(model, x_len[0:5], x[0:5], train_dataset.ix2w)
                vis.train_reconstruct(x_len[0:5], x[0:5], x_recon[0:5], train_dataset.ix2w)

        logger.info("\nepoch:{}".format(epoch))
        out = "--TRAIN-- running_loss:{:.3f}, kld:{:.3f}, bce:{:.3f}".format(running_loss, r_kldloss, r_bceloss)
        logger.info(out)
        model.eval()
        running_loss = 0.0
        r_bceloss = 0.0
        r_kldloss = 0.0

        for i, (x_len, x, y, eos_seq) in enumerate(valid_dataloader):
            # x_recon, x doesnt start with eos
            # y ends with eos 
            x_recon, q_mu, q_logvar = model(x_len, x, eos_seq)
            bce, kld = model.loss_fn(y, x_recon, q_mu, q_logvar, x_len)

            loss = bce/batch_size + kld/batch_size
            running_loss += loss.item()
            r_kldloss += (kld/batch_size).item()
            r_bceloss += (bce/batch_size).item()

        out = "--VALID-- running_loss:{:.3f}, kld:{:.3f}, bce:{:.3f}".format(running_loss, r_kldloss, r_bceloss)
        logger.info(out)


        if epoch%10==0 and epoch!=0:
            SAVEDTO = os.path.join(SAVE_PATH+"-{}.pt".format(str(epoch)))
            torch.save(model.state_dict(), SAVEDTO)
            print("saved to:", SAVEDTO)

#train_dloader = 
#for i, (x_len, x_pad, eos_seq) in enumerate(train_dloader):
#    output = model(x_pad, x_len, eos_seq)
#    loss += CrossEntropyLoss(output, target)

