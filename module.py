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
        s = 1
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
    