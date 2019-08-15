from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import pdb
import numpy as np
from nltk.tokenize import TweetTokenizer

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

        tokenizer = TweetTokenizer(preserve_case=False)
        for ix in range(len(self.data)):

            xx = self.data[ix]
            xx = tokenizer.tokenize(xx)[:self.max_seq_len]
        #    xx = xx.split()[:self.max_seq_len] # limit the length of sentences
            xx = [self.w2ix[w] if w in self.w2ix else self.w2ix['<unk>'] for w in xx]
            xx = torch.LongTensor(xx).to(self.device)
    
            sos2 = torch.LongTensor([2]).to(self.device)
            eos3 = torch.LongTensor([3]).to(self.device)

            #self.w2ix = {'<pad>':0, '<unk>':1, '<SOS>':2, '<EOS>': 3, 'N':4}

            ye = torch.cat((xx, eos3), dim=0)
            ey = torch.cat((sos2, xx), dim=0)
            data_.append((xx, ey, ye))
            

        self.data = data_
        #return train_seq, target_seq, EOS_seq


    def read_data(self, fn):
        with open(fn, 'r') as f:
            data = f.readlines()
        
        tokenizer = TweetTokenizer(preserve_case=False)
        
        data = [s.strip() for s in data]
        #all_words = [w for ws in data for w in ws.split()]
        all_words = [w for ws in data for w in tokenizer.tokenize(ws)]

        #all_words = [w for ws in tokenizer.tokenize(ws) in data]
        return data, all_words

    def make_ix_dicts(self, all_words):

        c_all_words = Counter(all_words)
        vocab_words = [w[0] for w in  c_all_words.items() if w[1]>4]
        #c_all_words = [w[0] for w in c_all_words

        print("all words:", len(vocab_words))

        #if len(c_all_words) > (self.nwords-5):
        #    vocab_words = c_all_words.most_common((self.nwords-5))
        #    vocab_words = [w[0] for w in vocab_words]
        #else:
        #    vocab_words = list(c_all_words.keys())

        vl = len(self.ix2w)
        for word in vocab_words:
            # special case for pctb
            if word == "<unk>" or word == "N":
                continue
            else:
                self.w2ix[word] = vl
                self.ix2w[vl] = word
                vl+=1

            if vl>=self.nwords:
                break

        #print(len(self.w2ix), "words")
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

def anneal(steps, anneal_steps):
    # logistic
    return float(1/(1+np.exp(-0.0025*(steps-anneal_steps))))


def drop_words(ey, y_lens, word_dropout):

    if word_dropout>0 and word_dropout<1:
        for i in range(ey.size(0)):
            prob = torch.rand(y_lens[i])
            if ey[i][0]==2:
                prob[0]=1 # keep sos
            #if ey[i][y_lens[i]]==3:
            #    prob[0]=1 # keep

            v = (prob<0.4).nonzero().flatten()
            ey[i][v] = 1  # drop 0.4 to unk
            #pdb.set_trace()
            #masks = np.random.choice(y_lens[i], int(y_lens[i]*word_dropout), replace=False)
            #ey[i][masks.tolist()] = 1 # <unk>

    if word_dropout==1:
        ey = torch.zeros_like(ey)

    return ey


