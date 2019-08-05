import torch
import pdb
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.manual_seed(0)

class RNNVAE(nn.Module):
    def __init__(self, z_dim=20, h_dim=100, nwords=5000, e_dim=300, framework="vae",
            rnngate="lstm", device='cpu', scale_pzvar=1):
        super(RNNVAE,self).__init__()
        self.embedding = nn.Embedding(nwords, e_dim, padding_idx=0)
        self.nwords = nwords
        self.encoder = RNNEncoder(z_dim=z_dim, 
                                    h_dim=h_dim, 
                                    e_dim=e_dim, rnngate=rnngate, device=device)
        self.decoder = RNNDecoder(z_dim=z_dim, 
                                    h_dim=h_dim, 
                                    e_dim=e_dim, nwords=nwords, rnngate=rnngate, device=device)
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.device = device
        self.rnngate = rnngate
        self.framework = framework
        self.scale_pzvar = scale_pzvar

        self.onehot_loss = nn.BCEWithLogitsLoss()
        self.bow_z_h = nn.Linear(z_dim, h_dim)
        self.bow_out = nn.Linear(h_dim, nwords)
 

    def sample_z_reparam(self, q_mu, q_logvar):
        eps = torch.randn_like(q_logvar)
        z = q_mu + torch.exp(q_logvar*0.5) * eps
        return z.to(self.device)

    def forward(self, xx, x_lens, ey, y_lens):

        embed = self.embedding(xx)
        q_mu, q_logvar = self.encoder(embed, x_lens)

        if self.framework=="vae":
            z = self.sample_z_reparam(q_mu, q_logvar)
        else:
            z = q_mu
        
        ey = self.embedding(ey)
        x_recon = self.decoder(y_lens, ey, z)

        return x_recon, z, q_mu, q_logvar

    def loss_fn(self, y, x_recon, q_mu, q_logvar):

        batch_ce_loss = 0.0
        for i in range(y.size(0)):
            ce_loss = F.cross_entropy(x_recon[i], y[i], reduction="mean", ignore_index=0)
            batch_ce_loss += ce_loss

        batch_ce_loss = batch_ce_loss/y.size(0)
        # check this?
        scale = 1/self.scale_pzvar
        kld = -0.5 * torch.sum(1 + q_logvar - q_mu.pow(2) - scale*q_logvar.exp())
        kld = kld/y.size(0)

        return batch_ce_loss, kld

    def loss_bow(self, y, z):
        predict = self.bow_out(F.relu(self.bow_z_h(z)))
        #predict = self.decoder.fc_out(F.relu(self.decoder.fc_z_h(z)))
        y_onehot = torch.zeros((y.size()[0], 10000)).to(self.device)
        for j in range(y.size()[0]):
            y_onehot[j][y[j]]=1

        
        loss = self.onehot_loss(predict.squeeze(0), y_onehot)
        loss = loss/y.size(0)
        #loss2 = nn.CrossEntropyLoss(predict.squeeze(0), y_onehot)
        #pdb.set_trace()
        return loss


class RNNEncoder(nn.Module):
    def __init__(self, z_dim=20, 
                        h_dim=100, 
                        n_layers=1, 
                        e_dim=300,
                        rnngate="lstm",
                        device='cpu'):
        super(RNNEncoder, self).__init__()

        self.n_layers = n_layers
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.rnn = getattr(nn, rnngate.upper())(e_dim, h_dim, n_layers, batch_first=True)

        self.rnngate = rnngate
        #self.gru = nn.GRU(embedding_dim, h_dim, n_layers, batch_first=True)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, embed, x_lengths, hidden=None):
        packed = pack_padded_sequence(embed, x_lengths, batch_first=True, enforce_sorted=False)

        if self.rnngate=="lstm":
            output_packed, (last_hidden, cell) = self.rnn(packed, hidden)
        else:
            output_packed, last_hidden = self.rnn(packed, hidden)
        #last_hidden = last_hidden.view(5, self.h_dim)
        q_mu = self.fc_mu(last_hidden)
        q_logvar = self.fc_logvar(last_hidden)

        return q_mu, q_logvar


class RNNDecoder(nn.Module):
    def __init__(self, z_dim=20,
                        h_dim=100,
                        e_dim=300,
                        n_layers=1,
                        nwords=5000,
                        rnngate="lstm",
                        device='cpu'):

        super(RNNDecoder, self).__init__()
        self.h_dim = h_dim
        self.nn = getattr(nn, rnngate.upper())(e_dim, h_dim, n_layers, batch_first=True)
        self.fc_z_h = nn.Linear(z_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, nwords)
        self.rnngate = rnngate
        self.device = device

    def forward(self, y_length, embed, z):

        hidden = self.fc_z_h(z)
        ccell = torch.randn_like(hidden)
        
        packed = pack_padded_sequence(embed, y_length, batch_first=True, enforce_sorted=False)

        if self.rnngate=="lstm":
            outputs, _ = self.nn(packed, (hidden, ccell ))
        else:
            outputs, _ = self.nn(packed, hidden)

        outputs, y_length = pad_packed_sequence(outputs, batch_first=True,
                total_length=max(y_length))
        outputs = self.fc_out(outputs)

       #outputs = F.logsoftmax(self.fc_out(outputs))
        return outputs

    def rollout_decode(self, input0, z, embedding):
        all_decoded = []
        z = z.to(self.device)
        hiddens = self.fc_z_h(z)

        for i in range(z.size(0)):
            decoded = []            
            
            if self.rnngate=="gru":
                output0, hidden = self.nn(input0.unsqueeze(0), hiddens[i].view(1, 1, self.h_dim))
            else:
                first_hidden = hiddens[i].view(1, 1, self.h_dim).to(self.device)
                ccell = torch.tanh(first_hidden).to(self.device)
                ccell = torch.randn(first_hidden.size()).to(self.device)
                output0, (hidden, cell) = self.nn(input0.unsqueeze(0), (first_hidden, ccell))

            output0 = torch.argmax(F.softmax(self.fc_out(output0).squeeze(), dim=0)).item()
            decoded.append(output0)

            output = output0

            while (len(decoded)<20):

                outputx = embedding(torch.LongTensor([output]).to(self.device))
                if self.rnngate=="gru":
                    output, hidden = self.nn(outputx.unsqueeze(0), hidden)
                else:
                    output, (hidden, cell) = self.nn(outputx.unsqueeze(0), (hidden, cell))

                output = torch.argmax(F.softmax(self.fc_out(output).squeeze(), dim=0)).item()
                decoded.append(output)

            all_decoded.append(decoded)
        return all_decoded

            



        
        
