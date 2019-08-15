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
argparser.add_argument('--test_fn', type=str)
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
log_test = logger_utils.get_nn_logger(mode="test", args=args)
log_sample = logger_utils.get_sample_logger(mode="train", args=args)

device = torch.device("cuda:{}".format(args.cuda) if int(args.cuda)>=0 else "cpu")
print("device:", device)

def init():
    train_dataset = utils.TextDataset(fn=args.train_fn, nwords=args.nwords, device=device,
            max_seq_len=args.max_seq_len, word_dropout=args.word_dropout)

    train_dataset.make_ix_dicts(train_dataset.all_words)
    train_dataset.proc_data()
    train_dataloader = utils.get_dataloader(train_dataset, batch_size=args.batch_size)

    valid_dataset = utils.TextDataset(fn=args.valid_fn, nwords=args.nwords, device=device,
            max_seq_len=args.max_seq_len, word_dropout=0)

    valid_dataset.w2ix = train_dataset.w2ix
    valid_dataset.ix2w = train_dataset.ix2w
    valid_dataset.vocab_size = train_dataset.vocab_size

    valid_dataset.proc_data()
    valid_dataloader = utils.get_dataloader(valid_dataset, batch_size=args.batch_size)

    test_dataset = utils.TextDataset(fn=args.test_fn, nwords=args.nwords, device=device,
            max_seq_len=args.max_seq_len, word_dropout=0)

    test_dataset.w2ix = train_dataset.w2ix
    test_dataset.ix2w = train_dataset.ix2w
    test_dataset.vocab_size = train_dataset.vocab_size

    test_dataset.proc_data()
    test_dataloader = utils.get_dataloader(test_dataset, batch_size=args.batch_size)


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

    # we want to run train, valid, loss all within the epoch

    epoch_valid_loss = []
    for epoch in range(start_e, end_e):
        print("epoch:", epoch)
        if steps>args.kl_anneal_steps:
            print("kl end")

        #### TRAIN
        model.train()

#        steps, optimizer, model = train_(train_dataloader, log_train, steps, model, optimizer, train_dataset.ix2w, epoch, args)
        steps = train_(train_dataloader, log_train, steps, model, optimizer, train_dataset.ix2w, epoch, args)

        #### SAVE
        if epoch%10==0 and epoch!=0:
            SAVEDTO = os.path.join(SAVE_PATH+"-{}.pt".format(str(epoch)))
            torch.save(model.state_dict(), SAVEDTO)
            print("saved to:", SAVEDTO)


        model.eval()

        ### VALID, TEST
        valid_lossD = testvalid_(valid_dataloader, log_valid, model, args, mode="valid")
        test_lossD = testvalid_(test_dataloader, log_test, model, args, mode="test")
        epoch_valid_loss.append(np.mean(valid_lossD['bce']))

        if len(epoch_valid_loss)>20:
            if epoch_valid_loss[-1]>epoch_valid_loss[-2]>epoch_valid_loss[-3]>epoch_valid_loss[-4]>epoch_valid_loss[-5]:
                print("validation falls for (5 iterations) at itr:{}".format(epoch))
                min_epoch = np.argmin(epoch_valid_loss)
                print("Min ppl @ epoch {}:{:.3f}".format(min_epoch, epoch_valid_loss[min_epoch]))
                sys.exit(0)



def get_loss_D():
    lossD = {}
    lossD['running'] = []
    lossD['bce'] = []
    lossD['kld'] = []
    lossD['bow'] = []
    lossD['ppl'] = []
    return lossD

def print_loss(logger, lossD):

    string = ""
    sorted_keys = ['running', 'bce', 'kld', 'bow']

    for k in sorted_keys:
        loss = np.mean(lossD[k])
        string += "\t{:.3f}".format(loss)

    ppl = np.exp(np.mean(lossD['bce']))
    string += "\t{:.3f}".format(ppl)

    logger.info(string)

    return string


def train_(train_dataloader, log_train, steps, model, optimizer, ix2w, epoch, args):
    lossD = get_loss_D()

    # for each batch
    for i, (xx, x_lens, ey, ye, y_lens) in enumerate(train_dataloader):
        steps +=1
        # i only drop words on train.
        ey = utils.drop_words(ey, y_lens, args.word_dropout)

        x_recon, z, q_mu, q_logvar = model(xx, x_lens, ey, y_lens)
        bce, kld = model.loss_fn(ye, y_lens, x_recon, q_mu, q_logvar)

        loss = bce

        if args.framework=="vae":
            lossD['kld'].append(kld.item())
            kl_weight = utils.anneal(steps, args.kl_anneal_steps)
            loss+=(kld*kl_weight)

        if args.bow:
            bow_loss = model.loss_bow(ye, z)
            loss += bow_loss
            lossD['bow'].append(bow_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossD['running'].append(loss.item())
        lossD['bce'].append(bce.item())
        #lossD['ppl'].append(np.exp(bce.item()))

    out = print_loss(log_train, lossD)
    print("--TRAIN--" + out)

        #### Reconstructions
    model.eval()
    input0 = model.embedding(torch.LongTensor([0]).to(device))
    z = torch.randn(5, args.z_dim)
    vis.sample_reconstruct(log_sample, epoch, model, input0, z, ix2w)
    vis.input_reconstruct(log_sample, model, x_lens[0:5], xx[0:5], input0, ix2w, device)
    vis.train_reconstruct(log_sample, x_lens[0:5], xx[0:5], x_recon[0:5], ix2w)
        ####
    return steps
#    return steps, optimizer, model

def testvalid_(valid_dataloader, log_valid, model, args, mode="valid"):
    lossD = get_loss_D()

    for i, (xx, x_lens, ey, ye, y_lens) in enumerate(valid_dataloader):

        ey = utils.drop_words(ey, y_lens, args.word_dropout)
        ### NO ROLLOUT
        x_recon, z, q_mu, q_logvar = model(xx, x_lens, ey, y_lens)
        bce, kld = model.loss_fn(ye, y_lens, x_recon, q_mu, q_logvar)

        ### ROLL OUT DECODE

        #embed = model.embedding(xx)
        #q_mu, q_logvar = model.encoder(embed, x_lens)
 
        #if model.framework=="vae":
        #    z = model.sample_z_reparam(q_mu, q_logvar)
        #else:
        #    z = q_mu
        #input0 = model.embedding(torch.LongTensor([0]).to(device))
        #_, decoded_score = model.decoder.rollout_decode(input0, z, model.embedding, max(y_lens))

        #bce, kld = model.loss_fn(ye, decoded_score, q_mu, q_logvar)

        loss = bce
        if args.framework=="vae":
            loss += kld
            lossD['kld'].append(kld.item())

        if args.bow:
            bow_loss = model.loss_bow(ye, z)
            loss += bow_loss
            lossD['bow'].append(bow_loss.item())

        lossD['running'].append(loss.item())
        lossD['bce'].append(bce.item())
    #    lossD['ppl'].append(np.exp(bce.item()))

    out = print_loss(log_valid,  lossD)
    print("--{}--".format(mode.upper()) + out)
    return lossD

if __name__=="__main__":
    init()


