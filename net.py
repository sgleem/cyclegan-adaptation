import torch
import torch.nn as nn
import torch.nn.functional as F

from module import ConvSample, Residual,Residual_Cat, ConvSample2D, ReLU, liGRU

class GRU_HMM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(GRU_HMM, self).__init__()
        input_dim = kwargs.get("input_dim", 120)
        hidden_dim = kwargs.get("hidden_dim", 465)
        num_layers = kwargs.get("num_layers", 5)
        output_dim = kwargs.get("output_dim", 1920)

        # self.GRU = nn.GRU(input_size = input_dim, hidden_size = hidden_dim, 
        #     num_layers=num_layers, dropout=0.2, bidirectional=True)
        self.GRU = liGRU(input_size = input_dim, hidden_size = hidden_dim, 
            num_layers=num_layers, bidirectional=True)
        self.HMM = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        h, _ = self.GRU(x)
        h = torch.squeeze(h, dim=1)
        out = self.HMM(h)
        out = F.log_softmax(out, dim=1)
        return out

class ivecNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ivecNN, self).__init__()
        ivec_dim = kwargs.get("ivec_dim", 100)
        hidden_dim = kwargs.get("hidden_dim", 512)
        out_dim = kwargs.get("out_dim", 120)

        self.MLP = nn.Sequential(
            nn.Linear(ivec_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

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
            nn.Linear(hidden_dim, output_dim)
        )
        self.logvar = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        h = self.mlp(x)
        m = self.mean(h)
        s = self.logvar(h)
        out = torch.exp(s/2) * torch.randn_like(m) + m
        
        return m, s, out

class VAE_Generator(nn.Module):
    """
    (N, T, 120) -> (N , 120 -> 240 -> 480, T) -> (N, 480+480 -> 480+480 -> 480+480 -> 480, T) -> (N, 480 -> 240 -> 120, T) -> (N, T, 120)
    ivec (N, 480)
    """
    def __init__(self, *args, **kwargs):
        super(VAE_Generator, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        aux_dim = kwargs.get("aux_dim", 480)
        self.downsample = nn.Sequential(
            ConvSample(inC=feat_dim, outC=feat_dim*2, k=3, s=1, p=1),
            ConvSample(inC=feat_dim*2, outC=feat_dim*4, k=3, s=1, p=1)
        )
        self.res = nn.ModuleList([
            # Residual(inC=feat_dim*4, hiddenC=feat_dim*8, k=3, s=1, p=1),
            # Residual(inC=feat_dim*4, hiddenC=feat_dim*8, k=3, s=1, p=1),
            # Residual(inC=feat_dim*4, hiddenC=feat_dim*8, k=3, s=1, p=1)
            Residual_Cat(mainC=feat_dim*4, auxC=aux_dim, hiddenC=feat_dim*8, k=3, s=1, p=1),
            Residual_Cat(mainC=feat_dim*4, auxC=aux_dim, hiddenC=feat_dim*8, k=3, s=1, p=1),
            Residual_Cat(mainC=feat_dim*4, auxC=aux_dim, hiddenC=feat_dim*8, k=3, s=1, p=1)
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
        spk = ivec.unsqueeze(dim=2).expand(-1, -1, 128)
        for r in self.res:
            # h = spk+h
            h = r(h, spk)
        h = self.upsample(h)
        h = h.permute(0, 2, 1)
        out = self.out(h)
        
        return out

class VAE_Discriminator(nn.Module):
    """
    (N, 128, 120) -> (N, 1, 128, 120) -> (N, 4, 64, 60) -> (N, 16, 32, 30) -> (N, 64, 16, 15) -> (N, 256, 8, 7) -> (N, 1024, 4, 3)
    -> (N, 128*120 -> 512 -> 512 -> 1 & 283)
    """
    def __init__(self, *args, **kwargs):
        super(VAE_Discriminator, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        frame_dim = kwargs.get("frame_dim", 128)
        hidden_dim = kwargs.get("hidden_dim", 512)
        spk_dim = kwargs.get("spk_dim", 283)
        self.spk_dim = spk_dim

        self.downsample = nn.Sequential(
            ConvSample2D(inC=1, outC=4, k=4, s=2, p=1),
            ConvSample2D(inC=4, outC=16, k=4, s=2, p=1),
            ConvSample2D(inC=16, outC=64, k=4, s=2, p=1),
            ConvSample2D(inC=64, outC=256, k=4, s=2, p=1),
            ConvSample2D(inC=256, outC=1024, k=4, s=2, p=1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(12288, hidden_dim),
            nn.LeakyReLU()
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

######### Translation VAE #########
class VariationalEncoder(nn.Module):
    """
    (N, T, 120) -> (N , 120 -> 240 -> 480 -> 480 -> 480 -> 480 -> 240 -> 120, T) -> (N, T, 120)
    """
    def __init__(self, *args, **kwargs):
        super(VariationalEncoder, self).__init__()
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
        self.mean = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Tanh()
        )
        self.std = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = self.cnn_encoder(x)
        h = h.permute(0, 2, 1)
        m = self.mean(h)
        s = self.std(h)
        
        return m, s

class VariationalDecoder(nn.Module):
    """
    (N, T, 120), (N, T, 120) => (N, T, 120) 
        -> (N , 120 -> 240 -> 480 -> 480 -> 480 -> 480 -> 240 -> 120, T) -> (N, T, 120)
    """
    def __init__(self, *args, **kwargs):
        super(VariationalDecoder, self).__init__()
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
        self.out = nn.Linear(feat_dim, feat_dim)
        
    def forward(self, m, s):
        x = torch.exp(s/2) * torch.randn_like(m) + m
        x = x.permute(0, 2, 1)
        h = self.cnn_encoder(x)
        h = h.permute(0, 2, 1)
        out = self.out(h)
        
        return out

class MeanTranslator(nn.Module):
    """
    (N, T, 120) + (N, 100) -> (N, T, 120) + (N, 1->T, 100) 
        -> (N, T, 220) -> (N, T, 220->512->512->120)
    """
    def __init__(self, *args, **kwargs):
        super(MeanTranslator, self).__init__()
        feat_dim = kwargs.get("feat_dim", 120)
        hidden_dim = kwargs.get("hidden_dim", 512)
        ivec_dim = kwargs.get("ivec_dim", 100)

        input_dim = feat_dim + ivec_dim

        self.x_var = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self, x, ivec):
        ivec = ivec.unsqueeze(dim=1).expand(-1, 128, -1)
        inp = torch.cat((x,ivec), dim=2)
        var = self.x_var(inp)
        sa_x = x + var
        return sa_x

