from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import pdb
import numpy as np

class TextDataset(Dataset):
    def __init__(self, fn="", nwords=5000, max_seq_len=10, device="cpu", word_dropout=0.0):
        self.nwords = nwords
        self.max_seq_len = max_seq_len
        self.device=device

        self.w2ix = {'<pad>':0, '<unk>':1, '<SOS>':2, '<EOS>': 3, 'N':4}
        self.ix2w = {v:k for k, v in self.w2ix.items()}

        self.data, self.all_words = self.read_data(fn)
        self.vocab_size = 0
        self.word_dropout = word_dropout
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.data[ix]

    def proc_data(self):
        
        data_ = []
        for ix in range(len(self.data)):

            xx = self.data[ix]
            xx = xx.split()[:self.max_seq_len] # limit the length of sentences
            xx = [self.w2ix[w] if w in self.w2ix else self.w2ix['<unk>'] for w in xx]
            xx = torch.LongTensor(xx).to(self.device)
    
            pad0 = torch.LongTensor([0]).to(self.device)
            eos3 = torch.LongTensor([3]).to(self.device)

            ye = torch.cat((xx, eos3), dim=0)
            ey = torch.cat((pad0, xx), dim=0)
            
            if self.word_dropout==0:
                pass
            else:
                masks = np.random.choice(len(ye), int(len(ye)*self.word_dropout), replace=False)
                ey[masks.tolist()] = 0 
            
            data_.append((xx, ey, ye))
            

        self.data = data_
        #return train_seq, target_seq, EOS_seq


    def read_data(self, fn):
        with open(fn, 'r') as f:
            data = f.readlines()
        
        data = [s.strip() for s in data]
        all_words = [w for ws in data for w in ws.split()]
        return data, all_words

    def make_ix_dicts(self, all_words):

        c_all_words = Counter(all_words)

        if len(c_all_words) > (self.nwords-5):
            vocab_words = c_all_words.most_common((self.nwords-5))
        else:
            vocab_words = list(c_all_words.keys())

        vl = len(self.ix2w)
        for word in c_all_words:
            # special case for ptb
            if word == "<unk>" or word == "N":
                continue
            else:
                self.w2ix[word] = vl
                self.ix2w[vl] = word
                vl+=1

            if vl>=self.nwords:
                break

        print(len(self.w2ix), "words")
        self.vocab_size = len(self.w2ix)


def get_dataloader(dataset, batch_size=5):

    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)

    return data_loader



def collate_fn(data):
    
    #data.sort(key=lambda x: len(x[0]), reverse=True) #large to small
    (xx, ey, ye) = zip(*data)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in ey]

    xx = pad_sequence(xx, batch_first=True, padding_value=0)
    ey = pad_sequence(ey, batch_first=True, padding_value=0)
    ye = pad_sequence(ye, batch_first=True, padding_value=0)

    return xx, x_lens, ey, ye, y_lens

def anneal(kld, steps, anneal_steps):
    # logistic
    if steps<anneal_steps:
        return kld*float(1/(1+np.exp(-0.0025*(steps-anneal_steps))))
    return kld
