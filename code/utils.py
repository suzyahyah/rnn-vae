from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import pdb

class TextDataset(Dataset):
    def __init__(self, fn="", model="vae", nwords=5000, max_seq_len=10):
        self.nwords = nwords
        self.model = model
        self.max_seq_len = max_seq_len

        self.w2ix = {'<pad>':0, '<unk>':1, '<SOS>':2, '<EOS>': 3, 'N':4}
        self.ix2w = {v:k for k, v in self.w2ix.items()}

        self.data, self.all_words = self.read_data(fn)
        self.vocab_size = 0
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        # this needs to be modified depending on the input output of the model
        sent = self.data[ix]
        sent = sent.split()[:self.max_seq_len] # limit the length of sentences
        sent = [self.w2ix[w] if w in self.w2ix else self.w2ix['<unk>'] for w in sent]

        if self.model=="vae":
            train_seq = sent
            #target_seq = 
            #target_seq = sent.append(self.w2ix['<EOS>'])

        train_seq = torch.LongTensor(train_seq)
        target_seq = torch.cat((train_seq, torch.LongTensor([3])), dim=0)
        #target_seq = torch.LongTensor(target_seq)

        EOS_seq = torch.LongTensor([self.w2ix['<EOS>']])

        return train_seq, target_seq, EOS_seq


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
    
    data.sort(key=lambda x: len(x[0]), reverse=True) #large to small
    train_seq, target_seq, eos_seq = zip(*data)
    train_seq_lens = [len(s) for s in train_seq]

    train_seq = pad_sequence(train_seq, batch_first=True, padding_value=0)
    target_seq = pad_sequence(target_seq, batch_first=True, padding_value=0)

    #train_packed = pack_padded_sequence(train_seq, train_seq_lens, batch_first=True)


    return train_seq_lens, train_seq, target_seq, eos_seq


    
