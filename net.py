import torch
import torch.nn as nn
import torch.nn.functional as F

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
