# Not developed yet
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        hidden_dim = kwargs.get("hidden_dim", 512)

        num_layers = kwargs.get("num_layers", 3)
        max_sequence = kwargs.get("num_layers", 3)

        self.encoder = nn.GRU(input_size = feat_dim, hidden_size = hidden_dim, 
            num_layers=num_layers, dropout=0.2, bidirectional=True)
        self.encoder_out = nn.Linear(hidden_dim, feat_dim)

        # (prev_out + prev_hidden) -> (softmax attention) * (encoder_sequence)
        # (1, batch, hidden_size) -> (batch, hidden_size)
        # (num_layers*2, batch, hidden_size) -> (batch, 6*hidden_size)
        # => (1, batch, hidden_size)
        self.attn_y = nn.Linear(hidden_size, max_sequence)
        self.attn_s = nn.Linear(num_layers*2, max_sequence)
        self.attn = nn.Sequential(
            nn.Linear(max_sequence, max_sequence),
            nn.Softmax(dim=1)
        )
        

        self.decoder = nn.GRU(input_size = feat_dim, hidden_size = hidden_dim, 
            num_layers=num_layers, dropout=0.2, bidirectional=True)
        self.decoder_out = nn.Linear(hidden_dim, feat_dim)
    
    def forward(self, x_seq, train=False):
        out_seq = None

        enc_seq, enc_state = self.encoder(x_seq)
        h = self.encoder_out(enc_seq) # (input_seq, batch, feat_dim)
        h = h.permute(1, 0, 2) # (batch, input_seq, feat_dim)

        sos = torch.zeros_like(enc_seq[0]).cuda()
        y = torch.unsqueeze(sos, 0) # (1, batch, hidden_dim)
        dec_state = enc_state # (num_layers*2, batch, hidden_dim)
        if train:
            inp_len = x_seq.size()[0]
            for inp_idx in range(inp_len):
                y = self.attn_y(y) # (1, batch, max_sequence)
                s = torch.sum(dec_state, dim=0) # (1, batch, hidden_dim)
                s = self.attn_s(s) # (1, batch, max_sequence)

                c = self.attn(y+s) # (1, batch, max_sequence)
                c = c[:inp_len] # (1, batch, input_seq)
                c = c.permute(1, 0, 2) # (batch, 1, input_seq)
                z = torch.bmm(c, h) # (batch, 1, feat_dim)
                z = z.permute(1, 0, 2) # (1, batch, feat_dim)
                
                y, dec_state = self.decoder(z, dec_state)
                out = self.decoder_out(y)
                out_seq = torch.cat((out_seq, out)) if out_seq is not None else out
        else:
            inp_len = x_seq.size()[0]
            while True:
                if 
                y = self.attn_y(y) # (1, batch, max_sequence)
                s = torch.sum(dec_state, dim=0) # (1, batch, hidden_dim)
                s = self.attn_s(s) # (1, batch, max_sequence)

                c = self.attn(y+s) # (1, batch, max_sequence)
                c = c[:inp_len] # (1, batch, input_seq)
                c = c.permute(1, 0, 2) # (batch, 1, input_seq)
                z = torch.bmm(c, h) # (batch, 1, feat_dim)
                z = z.permute(1, 0, 2) # (1, batch, feat_dim)
                
                y, dec_state = self.decoder(z, dec_state)
                out = self.decoder_out(y)
                out_seq = torch.cat((out_seq, out)) if out_seq is not None else out
