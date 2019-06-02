import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.normWxz = nn.BatchNorm1d(output_size)
        self.Whz = nn.Linear(output_size, output_size, bias=True)

        self.Wxh = nn.Linear(input_size, output_size, bias=True)
        self.normWxh = nn.BatchNorm1d(output_size)
        self.Whh = nn.Linear(output_size, output_size, bias=True)

    def forward(self, x, h_prev):
        batch_size = x.size()[0]

        if batch_size == 1:
            Zt = self.Wxz(x) + self.Whz(h_prev)
        else:
            Zt = self.normWxz(self.Wxz(x)) + self.Whz(h_prev)
        Zt = torch.sigmoid(Zt)

        if batch_size == 1:
            Ht = self.Wxh(x) + self.Whh(h_prev)
        else:
            Ht = self.normWxh(self.Wxh(x)) + self.Whh(h_prev)
        Ht = nn.ReLU()(Ht)

        h_out = Zt * h_prev + (1-Zt) * Ht
        return h_out

class liGRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(liGRU, self).__init__()
        input_size = kwargs.get("input_size", 0)
        hidden_size = kwargs.get("hidden_size", 0)
        num_layers = kwargs.get("num_layers", 0)
        self.bidirectional = kwargs.get("bidirectional", False)
        input_hidden = hidden_size*2 if self.bidirectional else hidden_size

        self.ligru_list = nn.ModuleList([])
        self.ligru_list.append(liGRUCell(input_size=input_size, output_size=hidden_size))
        if num_layers > 1:
            for cur_layer in range(1, num_layers):
                self.ligru_list.append(liGRUCell(input_size=input_hidden, output_size=hidden_size))
        
        self.init = torch.zeros(1, hidden_size).cuda()
        
    def forward(self, x, h_0=None):
        batch_size = x.size()[1]
        if h_0 is None:
            h_0 = self.init.expand(batch_size, -1)
        
        h_history = None
        for ligru in self.ligru_list:
            next_x = None
            h = h_0
            for cur_x in x:
                h_forward = ligru(cur_x, h)
                cur_h = torch.unsqueeze(h_forward, dim=0)
                next_x = cur_h if next_x is None else torch.cat((next_x, cur_h),dim=0)
                h = h_forward
            # save last hidden state
            h_history = cur_h if h_history is None else torch.cat((h_history, cur_h),dim=0)

            if self.bidirectional:
                back_x = None
                h = h_0
                x_back = torch.flip(x, dims=[0]) # sequence reverse
                for cur_x in x_back:
                    h_backward = ligru(cur_x, h)
                    cur_h = torch.unsqueeze(h_backward, dim=0)
                    back_x = cur_h if back_x is None else torch.cat((back_x, cur_h),dim=0)
                    h = h_backward
                # save last hidden state
                h_history = torch.cat((h_history, cur_h),dim=0)
                next_x = torch.cat((next_x, back_x),dim=2)

            x = next_x
        
        return x, h_history




