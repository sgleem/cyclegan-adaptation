import torch
import torch.nn as nn
import torch.nn.functional as F

from module import ConvSample, Residual,Residual_Cat, ConvSample2D, ReLU

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

        self.MLP = nn.Linear(feat_dim, feat_dim)
        self.MLP.weight = torch.nn.Parameter(torch.eye(feat_dim))
        self.MLP.bias = torch.nn.Parameter(torch.zeros(feat_dim))

    def forward(self, x):
        x_ = self.MLP(x)

        return x_

class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        hidden_dim = kwargs.get("hidden_dim", 512)
        num_layers = kwargs.get("num_layers", 2)

        self.MLP = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.PReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self, x):
        x = torch.squeeze(x)
        h = self.MLP(x)
        return h

class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        hidden_dim = kwargs.get("hidden_dim", 512)

        self.out = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        out = self.out(x)
        out = torch.sigmoid(out)
        
        return out

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
            Residual(inC=256, hiddenC=512, k=3, s=1, p=1)
        )
        self.upsample = nn.Sequential(
            ConvSample(inC=256, outC=128, k=5, s=1, p=2),
            ConvSample(inC=128, outC=feat_dim, k=5, s=1, p=2),
        )
        self.out = nn.Linear(feat_dim, feat_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h1 = self.downsample(x)
        h2 = self.res(h1)
        h3 = self.upsample(h2)
        h3 = h3.permute(0, 2, 1)
        out = self.out(h3)
        
        return out

class Discriminator_CNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Discriminator_CNN, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        hidden_dim = kwargs.get("hidden_dim", 512)
        self.downsample = nn.Sequential(
            ConvSample(inC=feat_dim, outC=128, k=5, s=1, p=2),
            ConvSample(inC=128, outC=256, k=5, s=1, p=2)
        )
        self.out = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = self.downsample(x)
        h = h.permute(0, 2, 1)
        out = self.out(h)
        out = torch.sigmoid(out)
        
        return out

######### Speaker normalizer #########

class VAE_Encoder(nn.Module):
    """
    (N, T, 120) -> (N , 120 -> 240 -> 480 -> 480 -> 480 -> 480 -> 240 -> 120, T) -> (N, T, 120)
    """
    def __init__(self, *args, **kwargs):
        super(VAE_Encoder, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        self.cnn_encoder = nn.Sequential(
            ConvSample(inC=feat_dim, outC=feat_dim*2, k=3, s=1, p=1),
            ConvSample(inC=feat_dim*2, outC=feat_dim*4, k=3, s=1, p=1),
            Residual(inC=feat_dim*4, hiddenC=feat_dim*8, k=3, s=1, p=1),
            Residual(inC=feat_dim*4, hiddenC=feat_dim*8, k=3, s=1, p=1),
            Residual(inC=feat_dim*4, hiddenC=feat_dim*8, k=3, s=1, p=1),
            ConvSample(inC=feat_dim*4, outC=feat_dim*2, k=3, s=1, p=1),
            ConvSample(inC=feat_dim*2, outC=feat_dim, k=3, s=1, p=1)
        )
        self.out = nn.Sequential(
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = self.cnn_encoder(x)
        h = h.permute(0, 2, 1)
        out = self.out(h)
        
        return out

class VAE_Speaker(nn.Module):
    """
    (N, 100 -> 512 -> 512 -> 480*2) 
    """
    def __init__(self, *args, **kwargs):
        super(VAE_Speaker, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        hidden_dim = kwargs.get("hidden_dim", 512)
        output_dim = kwargs.get("output_dim", 480)

        self.mlp = nn.Sequential(
            ReLU(input_dim=feat_dim, output_dim=hidden_dim, batch_norm=True, dropout=0),
            ReLU(input_dim=hidden_dim, output_dim=hidden_dim, batch_norm=True, dropout=0)
        )
        self.mean = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        self.logvar = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        h = self.mlp(x)
        m = self.mean(h)
        s = self.logvar(h)
        out = torch.exp(s) * torch.randn_like(m) + m
        
        return m, s, out

class VAE_Generator(nn.Module):
    """
    (N, T, 120) -> (N , 120 -> 240 -> 480, T) -> (N, 480+480 -> 480+480 -> 480+480 -> 480, T) -> (N, 480 -> 240 -> 120, T) -> (N, T, 120)
    ivec (N, 480)
    """
    def __init__(self, *args, **kwargs):
        super(VAE_Generator, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        self.downsample = nn.Sequential(
            ConvSample(inC=feat_dim, outC=feat_dim*2, k=3, s=1, p=1),
            ConvSample(inC=feat_dim*2, outC=feat_dim*4, k=3, s=1, p=1)
        )
        self.res = nn.ModuleList([
            # Residual(inC=feat_dim*4, hiddenC=feat_dim*8, k=3, s=1, p=1),
            # Residual(inC=feat_dim*4, hiddenC=feat_dim*8, k=3, s=1, p=1),
            # Residual(inC=feat_dim*4, hiddenC=feat_dim*8, k=3, s=1, p=1)
            Residual_Cat(inC=feat_dim*4, auxC=feat_dim*4, hiddenC=feat_dim*8, k=3, s=1, p=1),
            Residual_Cat(inC=feat_dim*4, auxC=feat_dim*4, hiddenC=feat_dim*8, k=3, s=1, p=1),
            Residual_Cat(inC=feat_dim*4, auxC=feat_dim*4, hiddenC=feat_dim*8, k=3, s=1, p=1)
        ])
        self.upsample = nn.Sequential(
            ConvSample(inC=feat_dim*4, outC=feat_dim*2, k=3, s=1, p=1),
            ConvSample(inC=feat_dim*2, outC=feat_dim, k=3, s=1, p=1)
        )
        self.out = nn.Sequential(
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, x, ivec):
        x = x.permute(0, 2, 1)
        h = self.downsample(x)
        spk = ivec.unsqueeze(dim=2).expand_as(h)
        for res in self.res:
            # h = spk+h
            h = res(h, spk)
        h = self.upsample(h)
        h = h.permute(0, 2, 1)
        out = self.out(h)
        
        return out

class VAE_Discriminator(nn.Module):
    """
    (N, 128, 120) -> (N, 1, 128, 120) -> (N, 4, 64, 60) -> (N, 16, 32, 30) -> (N, 64, 16, 15)
    -> (N, 128*120 -> 512 -> 512 -> 1 & 283)
    """
    def __init__(self, *args, **kwargs):
        super(VAE_Discriminator, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        hidden_dim = kwargs.get("hidden_dim", 512)
        spk_dim = kwargs.get("spk_dim", 283)
        self.spk_dim = spk_dim

        self.downsample = nn.Sequential(
            ConvSample2D(inC=1, outC=4, k=4, s=2, p=1),
            ConvSample2D(inC=4, outC=16, k=4, s=2, p=1),
            ConvSample2D(inC=16, outC=64, k=4, s=2, p=1)
        )
        self.mlp = nn.Sequential(
            ReLU(input_dim=feat_dim*128, output_dim=hidden_dim, batch_norm=False, dropout=0)
        )
        self.tf = nn.Linear(hidden_dim, 1)
        self.spk = nn.Linear(hidden_dim, spk_dim)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        h = self.downsample(x)
        h = torch.flatten(h, start_dim=1)
        h = self.mlp(h)
        tf = self.tf(h); tf = torch.sigmoid(tf)
        spk = self.spk(h); spk = F.log_softmax(spk, dim=1)
        
        return tf, spk
