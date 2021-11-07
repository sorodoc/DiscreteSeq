import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ninp, nhead, nhid, nlayers, nonlin, dropout=0.5, self_attn=1, query_attn=1, pos_encoding = 1):
        super(Transformer, self).__init__()
        try:
            from transformer_lib import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        encoder_layers = TransformerEncoderLayer(nhid, nhead, nhid, dropout, self_attn)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(ninp, nhid)
        self.w_emb = nn.Linear(ninp, nhid)
        self.ninp = ninp
        self.query_attn = query_attn
        self.positional = pos_encoding
        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        else:
            self.nonlin = nn.RReLU()
        #self.decoder = nn.Linear(ninp, ninp)
        self.out_hid = nn.Linear(nhid * 2, nhid)
        self.out_lin = nn.Linear(nhid, 1)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)

    def attention(self, inp, query):
        inp = torch.transpose(inp, 0, 1)
        weights = torch.bmm(inp, query.unsqueeze(2))
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        self.attn_weights = weights
        b_size, seq_size, hid_size = list(inp.size())
        return torch.bmm(torch.transpose(inp, 1, 2), weights).squeeze(2)

    def forward(self, src, query, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.encoder(src) * math.sqrt(self.ninp)
        if self.positional:
            src = self.pos_encoder(src)
        output, self_attn_w = self.transformer_encoder(src, self.src_mask)
        q_emb = self.w_emb(query)
        q_emb = self.nonlin(q_emb)
        if self.query_attn:
            comb_hid = torch.cat((self.attention(output, q_emb), q_emb), 1)
        else:
            comb_hid = torch.cat((output[-1], q_emb), 1)
            self.attn_weights = None
        out_hid = self.out_hid(comb_hid)
        out_nonlin = self.nonlin(out_hid)
        out = self.out_lin(out_nonlin)
        sig = torch.sigmoid(out.squeeze())
        return sig, self.attn_weights
