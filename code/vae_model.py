import torch
import pdb
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNVAE(nn.Module):
    def __init__(self, z_dim=20, hidden_dim=100, nwords=5000, embedding_dim=300):
        super(RNNVAE,self).__init__()
        self.embedding = nn.Embedding(nwords, embedding_dim)
        self.nwords = nwords
        self.encoder = RNNEncoder(z_dim, hidden_dim)
        self.decoder = RNNDecoder(z_dim, hidden_dim, nwords=nwords)
        self.z_dim = z_dim

    def sample_z_reparam(self, q_mu, q_logvar):
        eps = torch.randn_like(q_logvar)
        z = q_mu + torch.exp(q_logvar*0.5) * eps
        return z

    def forward(self, x_lengths, x_padded, eos_seq):

        embed = self.embedding(x_padded)
        q_mu, q_logvar = self.encoder(embed, x_lengths)
        z = self.sample_z_reparam(q_mu, q_logvar)
        
        eos_seq = self.embedding(torch.stack(eos_seq))
        eos_embed_seq = torch.cat((eos_seq, embed), dim=1)
        x_lengths = [(x+1) for x in x_lengths]

        x_recon = self.decoder(x_lengths, eos_embed_seq, z)

        return x_recon, q_mu, q_logvar

    def loss_fn(self, y, x_recon, q_mu, q_logvar, x_lengths):

        batch_ce_loss = 0.0
        for i in range(y.size(0)):
            # for each sentence
            x_len = x_lengths[i]+1 # becof of EOS
            ce_loss = F.cross_entropy(x_recon[i][:x_len], y[i][:x_len], reduction="sum")
            # should this be mean?
            #ce_loss = ce_loss/(x_len[i]+1) # normalize average word loss

        batch_ce_loss += ce_loss

        # check this?
        kld = -0.5 * torch.sum(1 + q_logvar - q_mu.pow(2) - q_logvar.exp())
        return batch_ce_loss, kld



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
                        nwords=5000):

        super(RNNDecoder, self).__init__()
        self.fc_z_h = nn.Linear(z_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        self.grucell = nn.GRUCell(embedding_dim, hidden_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, nwords)

    def forward(self, x_length, embed, z):

        hidden = self.fc_z_h(z)
        
        packed = pack_padded_sequence(embed, x_length, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, x_length = pad_packed_sequence(outputs, batch_first=True)
        outputs = self.fc_out(outputs)

       #outputs = F.logsoftmax(self.fc_out(outputs))
        return outputs

    def rollout_decode(self, input0, z, embedding):
        all_decoded = []
        hiddens = self.fc_z_h(z)

        for i in range(z.size(0)):
            decoded = []            

            output0, hidden = self.gru(input0.unsqueeze(0), hiddens[i].view(1, 1, self.hidden_dim))
            output0 = torch.argmax(F.softmax(self.fc_out(output0).squeeze(), dim=0)).item()
            decoded.append(output0)

            output = output0
            while (len(decoded)<20):

                outputx = embedding(torch.LongTensor([output]))
                output, hidden = self.gru(outputx.unsqueeze(0), hidden)
                output = torch.argmax(F.softmax(self.fc_out(output).squeeze(), dim=0)).item()
                decoded.append(output)

            all_decoded.append(decoded)
        return all_decoded

            



        
        
