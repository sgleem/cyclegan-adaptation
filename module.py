import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

class ConvSample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ConvSample, self).__init__()
        inC = kwargs.get("inC", 120)
        outC = kwargs.get("outC", 256)
        k = kwargs.get("k", 5)
        s = 1
        p = kwargs.get("p", 2)

        self.cnn = nn.Conv1d(inC, outC, kernel_size=k, stride=s, padding=p)
        self.cnn_norm = nn.InstanceNorm1d(outC)
        self.gate = nn.Conv1d(inC, outC, kernel_size=k, stride=s, padding=p)
        self.gate_norm = nn.InstanceNorm1d(outC)
    def forward(self, x):
        h1 = self.cnn(x); h1_norm = self.cnn_norm(h1)
        h2 = self.gate(x); h2_norm = self.gate_norm(h2)
        h2_sig = torch.sigmoid(h2_norm)

        out = h1_norm * h2_sig
        return out

class ConvSample2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ConvSample2D, self).__init__()
        inC = kwargs.get("inC", 1)
        outC = kwargs.get("outC", 16)
        k = kwargs.get("k", 3)
        s = kwargs.get("s", 1)
        p = kwargs.get("p", 1)

        self.cnn = nn.Conv2d(inC, outC, kernel_size=k, stride=s, padding=p)
        self.cnn_norm = nn.InstanceNorm2d(outC)
        self.gate = nn.Conv2d(inC, outC, kernel_size=k, stride=s, padding=p)
        self.gate_norm = nn.InstanceNorm2d(outC)
    def forward(self, x):
        h1 = self.cnn(x); h1_norm = self.cnn_norm(h1)
        h2 = self.gate(x); h2_norm = self.gate_norm(h2)
        h2_sig = torch.sigmoid(h2_norm)

        out = h1_norm * h2_sig
        return out

class Residual(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Residual, self).__init__()
        inC = kwargs.get("inC", 512)
        outC = kwargs.get("hiddenC", 1024)
        k = kwargs.get("k", 3)
        s = 1
        p = kwargs.get("p", 1)

        self.cnn1 = nn.Conv1d(inC, outC, kernel_size=k, stride=s, padding=p)
        self.cnn1_norm = nn.InstanceNorm1d(outC)
        self.gate = nn.Conv1d(inC, outC, kernel_size=k, stride=s, padding=p)
        self.gate_norm = nn.InstanceNorm1d(outC)

        self.cnn2 = nn.Conv1d(outC, inC, kernel_size=k, stride=s, padding=p)
        self.cnn2_norm = nn.InstanceNorm1d(inC)
    def forward(self, x):
        h1 = self.cnn1(x); h1_norm = self.cnn1_norm(h1)
        h2 = self.gate(x); h2_norm = self.gate_norm(h2)
        h2_sig = torch.sigmoid(h2_norm)
        glu = h1_norm * h2_sig

        h3 = self.cnn2(glu); h3_norm = self.cnn2_norm(h3)
        out = h3_norm + x
        return out

class Residual_Cat(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Residual_Cat, self).__init__()
        mainC = kwargs.get("mainC", 512)
        auxC = kwargs.get("auxC", 100)
        outC = kwargs.get("hiddenC", 1024)

        inC = mainC + auxC

        k = kwargs.get("k", 3)
        s = 1
        p = kwargs.get("p", 1)

        self.cnn1 = nn.Conv1d(inC, outC, kernel_size=k, stride=s, padding=p)
        self.cnn1_norm = nn.InstanceNorm1d(outC)
        self.gate = nn.Conv1d(inC, outC, kernel_size=k, stride=s, padding=p)
        self.gate_norm = nn.InstanceNorm1d(outC)

        self.cnn2 = nn.Conv1d(outC, mainC, kernel_size=k, stride=s, padding=p)
        self.cnn2_norm = nn.InstanceNorm1d(mainC)
    def forward(self, inX, auX):
        x = torch.cat((inX, auX), dim=1)
        h1 = self.cnn1(x); h1_norm = self.cnn1_norm(h1)
        h2 = self.gate(x); h2_norm = self.gate_norm(h2)
        h2_sig = torch.sigmoid(h2_norm)
        glu = h1_norm * h2_sig

        h3 = self.cnn2(glu); h3_norm = self.cnn2_norm(h3)
        out = h3_norm + inX
        return out

class ReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ReLU, self).__init__()
        input_dim = kwargs.get("input_dim", 120)
        output_dim = kwargs.get("output_dim", 1024)
        
        batch_norm = kwargs.get("batch_norm", True)
        dropout = kwargs.get("dropout", 0)

        self.linear = nn.Linear(input_dim, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if 0 < dropout < 1 else None
    def forward(self, x):
        h = self.linear(x)
        if self.batchnorm is not None:
            h = self.batchnorm(h)
        h = self.act(h)
        if self.dropout is not None:
            h = self.dropout(h)
        
        return h

class liGRUCell(nn.Module):
    def __init__(self, *args, **kwargs):
        super(liGRUCell, self).__init__()
        input_size = kwargs.get("input_size", 0)
        output_size = kwargs.get("output_size", 0)

        self.Wxz = nn.Linear(input_size, output_size, bias=True)
        self.Whz = nn.Linear(output_size, output_size, bias=True)
        self.normWxz = nn.BatchNorm1d(output_size)

        self.Wxh = nn.Linear(input_size, output_size, bias=True)
        self.normWxh = nn.BatchNorm1d(output_size)
        self.Whh = nn.Linear(output_size, output_size, bias=True)

    def forward(self, x, h):
        batch_size = x.size()[0]

        if batch_size == 1:
            Zt = self.Wxz(x) + self.Whz(h)
        else:
            Zt = self.normWxz(self.Wxz(x)) + self.Whz(h)
        Zt = torch.sigmoid(Zt)

        if batch_size == 1:
            Ht = self.Wxh(x) + self.Whh(h)
        else:
            Ht = self.normWxh(self.Wxh(x)) + self.Whh(h)
        Ht = F.relu(Ht)

        h_out = Zt * h + (1-Zt) * Ht
        return h_out

class liGRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(liGRU, self).__init__()
        input_size = kwargs.get("input_size", 0)
        hidden_size = kwargs.get("hidden_size", 0)
        num_layers = kwargs.get("num_layers", 0)
        self.bidirectional = kwargs.get("bidirectional", False)
        hidden_input = hidden_size * 2 if self.bidirectional else hidden_size

        self.ligru_list = nn.ModuleList([])
        self.ligru_list.append(liGRUCell(input_size=input_size, output_size=hidden_size))
        if num_layers > 1:
            for cur_layer in range(1, num_layers):
                self.ligru_list.append(liGRUCell(input_size=hidden_input, output_size=hidden_size))
        
        self.init_h = torch.zeros(1, hidden_size).cuda()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, x, h_0=None):
        x.contiguous()
        max_seq = x.size()[0]
        batch_size = x.size()[1]

        if self.bidirectional:
            # initial zero state
            if h_0 is None:
                h_0 = self.init_h.expand(batch_size * 2, -1)        
        else:
            # initial zero state
            if h_0 is None:
                h_0 = self.init_h.expand(batch_size, -1)

        # initialize hidden state history
        inp_x = x
        h_history = None
        # h_history = torch.empty(self.num_layers, max_seq, batch_size, self.hidden_size).cuda()
        for layer_idx in range(self.num_layers):
            # bidirectional -> add reversed sequence as different batch
            if self.bidirectional:
                # (max_seq, batch, feat_dim) -> (max_seq, 2 * batch, feat_dim)
                back_x = torch.flip(inp_x, [0])
                inp_x = torch.cat([inp_x, back_x], dim=1)
            ligru = self.ligru_list[layer_idx]
            cur_history = []
            # Forward calculation
            h_prev = h_0
            for seq_idx, cur_x in enumerate(inp_x):
                # (batch_size, input_dim) + (batch_size, hidden_dim) -> (batch_size, hidden_dim)
                h_cur = ligru(cur_x, h_prev)
                # save state to state history
                cur_history.append(h_cur)
                # save current state as next input state
                h_prev = h_cur
            # save curren state sequence as next inp
            next_x = torch.stack(cur_history)
            next_x.contiguous()
            forward_x = next_x[:, 0:batch_size]
            backward_x = next_x[:, batch_size:]
            next_x = torch.cat([forward_x, backward_x], dim=2)

            inp_x = next_x
            # h_history[layer_idx] = inp_x
        # unpack bidirectional
        # h_history = h_history.view(self.num_layers, max_seq, batch_size//2, self.hidden_size * 2)
        
        x = inp_x

        return x, h_history

def SDPA(self, Q, K=None, V=None):
        if K is None:
            K = Q
        if V is None:
            V = K
        
        # Q(N, 1, 120) * K(N, T, 120) => (N, 1, T)
        Q = Q.permute(1, 0, 2).contiguous(); K = K.permute(1, 0, 2).contiguous(); V = V.permute(1, 0, 2).contiguous()

        feat_dim = Q.size()[2]
        K_T = K.permute(0, 2, 1).contiguous()
        attn = torch.bmm(Q, K_T)
        attn = attn / np.sqrt(feat_dim)

        # Apply Softmax
        attn = F.softmax(attn, dim=1)

        # Attn (N, 1 or T, T) * V(N, T, 120) -> (N, 1 or T, 120)
        out = torch.bmm(attn, V)
        out = out.permute(1, 0, 2)
        return out

class MultiHeadAttention(nn.Module):
    # Q (previous feature) (T or 1, N, 120)
    # K (input sequence) (T, N, 120)
    # V (input sequence)
    def __init__(self, *args, **kwargs):
        super(MultiHeadAttention, self).__init__()
        input_size = kwargs.get("input_size", 120)
        hidden_size = kwargs.get("hidden_size", 512)
        n_head = kwargs.get("n_head", 8)
        self.Wq_set = nn.ModuleList([])
        self.Wk_set = nn.ModuleList([])
        self.Wv_set = nn.ModuleList([])
        for h in range(n_head):
            self.Wq_set.append(nn.Linear(input_size, hidden_size))
            self.Wk_set.append(nn.Linear(input_size, hidden_size))
            self.Wv_set.append(nn.Linear(input_size, hidden_size))
        
        self.out = nn.Linear(hidden_size * n_head, input_size)
        self.norm = nn.LayerNorm(input_size, elementwise_affine=False)
        self.n_head = n_head
    def forward(self, Q, K=None, V=None):
        if K is None:
            K = Q
        if V is None:
            V = K
        h = []
        for idx in range(self.n_head):
            curQ = self.Wq_set[idx](Q)
            curK = self.Wk_set[idx](K)
            curV = self.Wv_set[idx](V)
            cur_sdpa = SDPA(curQ, curK, curV) # (1 or T, N, 512)
            h.append(cur_sdpa)
        h = torch.cat(h, dim=2) # (1 or T, N, 512 * 8)
        result = self.out(h) # (1 or T, N, 120)
        # result = result + V # (1 or T, N, 120)
        # result = self.norm(result)
        return result

class FFN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FFN, self).__init__()
        input_size = kwargs.get("input_size", 120)
        hidden_size = kwargs.get("hidden_size", 1024)

        self.relu = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_size, input_size)
        self.norm = nn.LayerNorm(input_size, elementwise_affine=False)
    def forward(self, x):
        h = self.relu(x)
        out = self.out(h)

        # out = out + x
        # out = self.norm(out)

        return out




