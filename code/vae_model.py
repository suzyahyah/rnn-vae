import torch
import pdb
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNVAE(nn.Module):
    def __init__(self, z_dim=20, hidden_dim=100, nwords=10000, embedding_dim=300):
        super(RNNVAE,self).__init__()
        self.embedding = nn.Embedding(nwords, embedding_dim)
        self.encoder = RNNEncoder(z_dim, hidden_dim)
        self.decoder = RNNDecoder(z_dim, hidden_dim)
        self.z_dim = z_dim

    def sample_z_reparam(self, q_mu, q_logvar):
        eps = torch.randn_like(q_logvar)
        z = q_mu + torch.exp(q_logvar*0.5) * eps
        return z

    def forward(self, x_padded, x_lengths, eos_seq):

        embed = self.embedding(x_padded)
        q_mu, q_logvar = self.encoder(embed, x_lengths)
        z = self.sample_z_reparam(q_mu, q_logvar)
        
        eos_seq = self.embedding(torch.stack(eos_seq))
        x_recon = self.decoder(x_lengths, embed, eos_seq, z)

        return x_recon, q_mu, q_logvar

    def loss_fn(self, x, x_recon, q_mu, q_logvar):
        pass



class RNNEncoder(nn.Module):
    def __init__(self, z_dim=20, 
                        hidden_dim=100, 
                        n_layers=1, 
                        embedding_dim=300):
        super(RNNEncoder, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, embed, x_lengths, hidden=None):
        packed = pack_padded_sequence(embed, x_lengths, batch_first=True)
        output_packed, last_hidden = self.gru(packed, hidden)
        #last_hidden = last_hidden.view(5, self.hidden_dim)
        q_mu = self.fc_mu(last_hidden)
        q_logvar = self.fc_logvar(last_hidden)

        return q_mu, q_logvar


class RNNDecoder(nn.Module):
    def __init__(self, z_dim=20,
                        hidden_dim=100,
                        embedding_dim=300,
                        n_layers=1,
                        n_words=10000):

        super(RNNDecoder, self).__init__()
        self.fc_z_h = nn.Linear(z_dim, hidden_dim)

        self.grucell = nn.GRUCell(embedding_dim, hidden_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, n_words)

    def forward(self, x_length, embed, eos_seq, z):

        hidden = self.fc_z_h(z)
        
        #pdb.set_trace()
        #hidden = self.grucell(eos_seq, hidden)
        embed = torch.cat((eos_seq, embed), dim=1)
        packed = pack_padded_sequence(embed, x_length, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs = pad_packed_sequence(outputs, batch_first=True)
        #outputs = F.logsoftmax(self.fc_out(outputs))
        return outputs


        
        
