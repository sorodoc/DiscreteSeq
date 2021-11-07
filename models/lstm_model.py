import torch
import torch.nn as nn
import torch.nn.functional as F

class Lstm(nn.Module):
    def __init__(self, seq_inp_size, q_inp_size, nhid, rnn_type, nlayers, nonlin, attn_lstm = 0, dropout = 0.0):
        super(Lstm, self).__init__()
        self.seq_inp_size = seq_inp_size
        self.q_inp_size = q_inp_size
        self.nhid = nhid
        self.nlayers = nlayers
        self.seq_emb = nn.Linear(self.seq_inp_size, self.nhid)
        self.q_emb = nn.Linear(self.q_inp_size, self.nhid)
        self.out_hid = nn.Linear(self.nhid * 2, self.nhid)
        self.out_lin = nn.Linear(self.nhid, 1)
        self.attn_lstm = attn_lstm
        self.debug = 0
        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        else:
            self.nonlin = nn.RReLU()
        self.query_nonlin = self.nonlin
        self.rnn_type = rnn_type
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(seq_inp_size, nhid, self.nlayers, dropout = dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for '--model' was supplied """)
            self.rnn = nn.RNN(seq_inp_size, nhid, self.nlayers, nonlinearity = nonlinearity, dropout = dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.seq_emb.weight.data.uniform_(-initrange, initrange)
        self.q_emb.weight.data.uniform_(-initrange, initrange)

    def attention(self, rnn_out, query):
        rnn1 = torch.transpose(rnn_out,0, 1)
        weights = torch.bmm(rnn1, query.unsqueeze(2))
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        self.attn_weights = weights
        return torch.bmm(torch.transpose(rnn1, 1, 2), weights).squeeze(2)

    def forward(self, inp, hidden, query):
        q_emb = self.q_emb(query)
        out, hidden = self.rnn(inp, hidden)
        if self.attn_lstm:
            comb_hid = torch.cat((self.attention(out, q_emb), q_emb), 1)
        else:
            self.attn_weights = None
            if self.rnn_type == 'LSTM':
                comb_hid = torch.cat((hidden[0][-1], q_emb), 1)
            else:
                comb_hid = torch.cat((hidden[-1], q_emb), 1)
        out_hid = self.out_hid(comb_hid)
        out_nonlin = self.nonlin(out_hid)
        out = self.out_lin(out_nonlin)
        sig = torch.sigmoid(out.squeeze())
        return sig, hidden, self.attn_weights

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
