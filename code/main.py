import utils
import vae_model
import numpy as np
import torch
from torch import optim
import os
import pdb
import vis
import logging
import logger_utils
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--cuda', dest='cuda', type=str)
argparser.add_argument('--num_epochs', dest='num_epochs', type=int)
argparser.add_argument('--batch_size', dest='batch_size', type=int)
argparser.add_argument('--nwords', dest='nwords', type=int)
argparser.add_argument('--l_epoch', dest='l_epoch', type=int)
argparser.add_argument('--variational', dest='variational', type=int)
argparser.add_argument('--hidden_dim', dest='h_dim', type=int)
argparser.add_argument('--latent_dim', dest='z_dim', type=int)

args = argparser.parse_args()

np.random.seed(0)
if args.variational==1:
    fol = "rnn"
else:
    fol = "vae"

SAVE_PATH="models/{}/z{}-h{}-bs{}/model".format(fol, args.z_dim, args.h_dim, args.batch_size)


log_train = logger_utils.get_nn_logger(mode="train", args=args)
log_valid = logger_utils.get_nn_logger(mode="valid", args=args)
log_sample = logger_utils.get_sample_logger(args)

device = torch.device("cuda:{}".format(args.cuda) if int(args.cuda)>=0 else "cpu")
print("device:", device)


if __name__=="__main__":

    train_dataset = utils.TextDataset(fn="data/ptb.train.txt", nwords=args.nwords)
    train_dataset.make_ix_dicts(train_dataset.all_words)
    train_dataloader = utils.get_dataloader(train_dataset, batch_size=args.batch_size)

    valid_dataset = utils.TextDataset(fn="data/ptb.valid.txt", nwords=args.nwords)
    valid_dataset.make_ix_dicts(valid_dataset.all_words)
    valid_dataloader = utils.get_dataloader(valid_dataset, batch_size=args.batch_size)

    model = vae_model.RNNVAE(nwords=train_dataset.vocab_size, 
                                variational=args.variational, 
                                z_dim=args.z_dim,
                                h_dim=args.h_dim,
                                device=device)
    try:
        model.load_state_dict(torch.load(os.path.join(SAVE_PATH+"-{}.pt".format(str(args.l_epoch))),map_location=device))
        print("loaded from:", args.l_epoch)
        start_e = args.l_epoch
        end_e = args.l_epoch+args.num_epochs
    except:
        start_e = 0
        end_e = args.num_epochs
        print("failed to load from epoch:", args.l_epoch)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(start_e, end_e):
        print("epoch:", epoch)
        running_loss = []
        r_bceloss = []
        r_kldloss = []

        model.train()
        for i, (x_len, x, y, eos_seq) in enumerate(train_dataloader):
            # x_recon, x doesnt start with eos
            # y ends with eos 
            eos_seq = torch.stack(eos_seq)
            x = x.to(device)
            y = y.to(device)
            eos_seq = eos_seq.to(device)

            x_recon, q_mu, q_logvar = model(x_len, x, eos_seq)
            bce, kld = model.loss_fn(y, x_recon, q_mu, q_logvar)

            #loss = bce/args.batch_size # + kld/args.batch_size
            loss = bce/args.batch_size + kld/args.batch_size
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            running_loss.append(loss.item())
            r_kldloss.append((kld/args.batch_size).item())
            r_bceloss.append((bce/args.batch_size).item())

        model.eval()
        # Reconstructions
        input0 = model.embedding(torch.LongTensor([3]).to(device))
        z = torch.randn(5, args.z_dim)
        ix2w = train_dataset.ix2w
        vis.sample_reconstruct(log_sample, epoch, model, input0, z, ix2w)
        vis.input_reconstruct(log_sample, model, x_len[0:5], x[0:5], input0, ix2w, device)
        vis.train_reconstruct(log_sample, x_len[0:5], x[0:5], x_recon[0:5], ix2w)

        out = "{}\t{:.3f}\t{:.3f}\t{:.3f}".format(epoch, np.mean(running_loss),
                np.mean(r_kldloss), np.mean(r_bceloss))
        log_train.info(out)
        print("--TRAIN--" + out)

        running_loss = []
        r_bceloss = []
        r_kldloss = []


        for i, (x_len, x, y, eos_seq) in enumerate(valid_dataloader):
            # x_recon, x doesnt start with eos
            # y ends with eos 
            eos_seq = torch.stack(eos_seq)
            x = x.to(device)
            y = y.to(device)
            eos_seq = eos_seq.to(device)

            x_recon, q_mu, q_logvar = model(x_len, x, eos_seq)
            bce, kld = model.loss_fn(y, x_recon, q_mu, q_logvar)

            loss = bce/args.batch_size + kld/args.batch_size
            running_loss.append(loss.item())
            r_kldloss.append((kld/args.batch_size).item())
            r_bceloss.append((bce/args.batch_size).item())
 
        out = "{}\t{:.3f}\t{:.3f}\t{:.3f}".format(epoch, np.mean(running_loss),
                np.mean(r_kldloss), np.mean(r_bceloss))
 
        log_valid.info(out)
        print("--VALID--" + out)

        if epoch%100==0 and epoch!=0:
            SAVEDTO = os.path.join(SAVE_PATH+"-{}.pt".format(str(epoch)))
            torch.save(model.state_dict(), SAVEDTO)
            print("saved to:", SAVEDTO)


