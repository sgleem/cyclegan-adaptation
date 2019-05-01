import torch
import torch.nn as nn
import torch.nn.functional as F

from module import ConvSample, Residual

class GRU_HMM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(GRU_HMM, self).__init__()
        input_dim = kwargs.get("input_dim", 120)
        hidden_dim = kwargs.get("hidden_dim", 465)
        num_layers = kwargs.get("num_layers", 5)
        output_dim = kwargs.get("output_dim", 1920)

        self.GRU = nn.GRU(input_size = input_dim, hidden_size = hidden_dim, 
            num_layers=num_layers, dropout=0.2, bidirectional=True)
        self.HMM = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        h, _ = self.GRU(x)
        h = torch.squeeze(h, dim=1)
        out = self.HMM(h)
        out = F.log_softmax(out, dim=1)
        return out

class FDLR(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FDLR, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        hidden_dim = kwargs.get("hidden_dim", 512)
        num_layers = kwargs.get("num_layers", 2)

        self.GRU = nn.GRU(input_size = feat_dim, hidden_size = hidden_dim, num_layers=num_layers, dropout=0.2)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.PReLU(),
            nn.Linear(int(hidden_dim/2), feat_dim)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        h, _ = self.GRU(x)
        h = torch.squeeze(h, dim=1)
        x_ = self.out(h)

        return x_

class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        hidden_dim = kwargs.get("hidden_dim", 512)
        num_layers = kwargs.get("num_layers", 2)

        self.GRU = nn.GRU(input_size = feat_dim, hidden_size = hidden_dim, num_layers=num_layers, dropout=0.2)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.PReLU(),
            nn.Linear(int(hidden_dim/2), feat_dim)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        h, _ = self.GRU(x)
        h = torch.squeeze(h, dim=1)
        x_ = self.out(h)
        return x_

class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        hidden_dim = kwargs.get("hidden_dim", 512)

        self.MLP = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        h = self.MLP(x)
        prob = self.out(h)
        return prob

class Generator_CNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Generator_CNN, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        num_down = kwargs.get("num_down", 2)
        num_res = kwargs.get("num_res", 6)
        num_up = kwargs.get("num_up", 2)

        self.downsample = nn.Sequential(
            ConvSample(inC=feat_dim, outC=128, k=5, s=1, p=2),
            ConvSample(inC=128, outC=256, k=5, s=1, p=2)
        )
        self.res = nn.Sequential(
            Residual(inC=256, hiddenC=512, k=3, s=1, p=1),
            Residual(inC=256, hiddenC=512, k=3, s=1, p=1),
            Residual(inC=256, hiddenC=512, k=3, s=1, p=1),
            Residual(inC=256, hiddenC=512, k=3, s=1, p=1),
            Residual(inC=256, hiddenC=512, k=3, s=1, p=1),
            Residual(inC=256, hiddenC=512, k=3, s=1, p=1)
        )
        self.upsample = nn.Sequential(
            ConvSample(inC=256, outC=128, k=5, s=1, p=2),
            ConvSample(inC=128, outC=feat_dim, k=5, s=1, p=2)
        )
        self.out = nn.Linear(feat_dim, feat_dim)

    def forward(self, x):
        decode = True if x.dim()==2 else False
        if decode:
            x = x.permute(1, 0)
        else:
            x = x.permute(0, 2, 1)
        h1 = self.downsample(x)
        h2 = self.res(h1)
        h3 = self.upsample(h2)
        if decode:
            h3 = h3.permute(1, 0)
        else:
            h3 = h3.permute(0, 2, 1)
        out = self.out(h3)
        
        return out

class Discriminator_CNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Discriminator_CNN, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        num_down = kwargs.get("num_down", 3)

        self.downsample = nn.Sequential(
            ConvSample(inC=feat_dim, outC=128, k=7, s=1, p=3),
            ConvSample(inC=128, outC=256, k=5, s=1, p=2),
            ConvSample(inC=256, outC=512, k=3, s=1, p=1),
        )
        self.out = nn.Linear(512, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = self.downsample(x)
        h = h.permute(0, 2, 1)
        out = self.out(h)
        out = torch.sigmoid(out)
        
        return out
