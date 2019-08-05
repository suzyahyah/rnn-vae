import utils
import vae_model
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_value_
import os
import pdb
import vis
import logging
import logger_utils
import argparse
from distutils.util import strtobool
import sys

torch.manual_seed(0)
np.random.seed(0)

argparser = argparse.ArgumentParser()
# filepaths
argparser.add_argument('--train_fn', type=str)
argparser.add_argument('--valid_fn', type=str)
argparser.add_argument('--old_emb', type=str)
argparser.add_argument('--new_emb',  type=str)
argparser.add_argument('--vocab_fn', type=str)
# mode
argparser.add_argument('--cuda', dest='cuda', type=str, default="-1")
argparser.add_argument('--num_epochs', dest='num_epochs', type=int)
argparser.add_argument('--batch_size', dest='batch_size', type=int)
argparser.add_argument('--nwords', dest='nwords', type=int)
argparser.add_argument('--l_epoch', dest='l_epoch', type=int)
argparser.add_argument('--framework', dest='framework', type=str)
argparser.add_argument('--hidden_dim', dest='h_dim', type=int)
argparser.add_argument('--latent_dim', dest='z_dim', type=int)
argparser.add_argument('--embedding_dim', dest='e_dim', type=int)
argparser.add_argument('--rnngate', dest='rnngate', type=str)
argparser.add_argument('--n_layers', dest='n_layers', type=int)
argparser.add_argument('--max_seq_len', dest='max_seq_len', type=int, default=25)
argparser.add_argument('--delta_weight', type=lambda x: bool(strtobool(x)), default=False)
argparser.add_argument('--universal_embed', type=lambda x: bool(strtobool(x)), default=False)
argparser.add_argument('--bow', type=lambda x: bool(strtobool(x)), default=False)
argparser.add_argument('--z_combine', type=str)
argparser.add_argument('--word_dropout', type=float, default=0.0)
argparser.add_argument('--scale_pzvar', type=float, default=1.0)
argparser.add_argument('--kl_anneal_steps', type=int, default=0)

args = argparser.parse_args()

SAVE_PATH = logger_utils.get_save_dir(args)

log_train = logger_utils.get_nn_logger(mode="train", args=args)
log_valid = logger_utils.get_nn_logger(mode="valid", args=args)
log_sample = logger_utils.get_sample_logger(mode="train", args=args)

device = torch.device("cuda:{}".format(args.cuda) if int(args.cuda)>=0 else "cpu")
print("device:", device)


if __name__=="__main__":

    train_dataset = utils.TextDataset(fn=args.train_fn, nwords=args.nwords, device=device,
            max_seq_len=args.max_seq_len, word_dropout=args.word_dropout)
    
    train_dataset.make_ix_dicts(train_dataset.all_words)
    train_dataset.proc_data()
    train_dataloader = utils.get_dataloader(train_dataset, batch_size=args.batch_size)

    valid_dataset = utils.TextDataset(fn=args.valid_fn, nwords=args.nwords, device=device,
            max_seq_len=args.max_seq_len, word_dropout=args.word_dropout)

    valid_dataset.w2ix = train_dataset.w2ix
    valid_dataset.ix2w = train_dataset.ix2w
    valid_dataset.vocab_size = train_dataset.vocab_size

    valid_dataset.proc_data()
    valid_dataloader = utils.get_dataloader(valid_dataset, batch_size=args.batch_size)

    model = vae_model.RNNVAE(nwords=train_dataset.vocab_size, 
                                framework=args.framework, 
                                z_dim=args.z_dim,
                                h_dim=args.h_dim,
                                e_dim=args.e_dim,
                                rnngate=args.rnngate,
                                device=device,
                                scale_pzvar=args.scale_pzvar)

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

    steps = 0

    epoch_valid_loss = []
    for epoch in range(start_e, end_e):
        print("epoch:", epoch)
        running_loss = []
        r_bceloss = []
        r_kldloss = []
        if steps>args.kl_anneal_steps:
            print("kl end")

        model.train()
        for i, (xx, x_lens, ey, ye, y_lens) in enumerate(train_dataloader):
            steps +=1 
            # x_recon, x doesnt start with eos
            # y ends with eos 
            x_recon, z, q_mu, q_logvar = model(xx, x_lens, ey, y_lens)
            bce, kld = model.loss_fn(ye, x_recon, q_mu, q_logvar)

            loss = bce 
            if args.framework=="vae":
                loss += utils.anneal(kld, steps, args.kl_anneal_steps)
            if args.bow:
                bow_loss = model.loss_bow(ye, z)
                loss += bow_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())
            r_kldloss.append(kld.item())
            r_bceloss.append(bce.item())

        model.eval()
        # Reconstructions
        input0 = model.embedding(torch.LongTensor([0]).to(device))
        z = torch.randn(5, args.z_dim)
        ix2w = train_dataset.ix2w
        vis.sample_reconstruct(log_sample, epoch, model, input0, z, ix2w)
        vis.input_reconstruct(log_sample, model, x_lens[0:5], xx[0:5], input0, ix2w, device)
        vis.train_reconstruct(log_sample, x_lens[0:5], xx[0:5], x_recon[0:5], ix2w)

        out = "{}\t{:.3f}\t{:.3f}\t{:.3f}".format(epoch, np.mean(running_loss),
                np.mean(r_kldloss), np.mean(r_bceloss))
        log_train.info(out)
        print("--TRAIN--" + out)
        print("Perplexity:", np.exp(np.mean(r_bceloss)))

        running_loss = []
        r_bceloss = []
        r_kldloss = []

        for i, (xx, x_lens, ey, ye, y_lens) in enumerate(valid_dataloader):
            # x_recon, x doesnt start with eos
            # y ends with eos 
            x_recon, z, q_mu, q_logvar = model(xx, x_lens, ey, y_lens)
            bce, kld = model.loss_fn(ye, x_recon, q_mu, q_logvar)
            valid_loss = bce + kld
            running_loss.append(valid_loss.item())
            r_kldloss.append(kld.item())
            r_bceloss.append(bce.item())
           
 
        out = "{}\t{:.3f}\t{:.3f}\t{:.3f}".format(epoch, np.mean(running_loss),
                np.mean(r_kldloss), np.mean(r_bceloss))
 
        log_valid.info(out)
        print("--VALID--" + out)
        print("Perplexity:", np.exp(np.mean(r_bceloss)))

        if epoch%10==0 and epoch!=0:
            SAVEDTO = os.path.join(SAVE_PATH+"-{}.pt".format(str(epoch)))
            torch.save(model.state_dict(), SAVEDTO)
            print("saved to:", SAVEDTO)


        epoch_valid_loss.append(np.mean(running_loss))
        if len(epoch_valid_loss)>6:
            if (max(epoch_valid_loss[-5:])==epoch_valid_loss[-1]):
                print("validation falls for (5 iterations) at itr:{}".format(epoch))
                min_epoch = np.argmin(epoch_valid_loss)
                print("Min loss @ epoch {}:{:.3f}".format(min_epoch,
                    epoch_valid_loss[min_epoch]))
                sys.exit(0)
