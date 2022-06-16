import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=128, bottle_dim=20) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim//2)
        self.fc2 = nn.Linear(latent_dim//2, latent_dim//4)
        self.mean = nn.Linear(latent_dim//4, bottle_dim)
        self.log_var = nn.Linear(latent_dim//4, bottle_dim)

    def kl_loss(self, mean, log_var):
        return (-0.5 * (1 + log_var - mean**2 - log_var.exp()).sum(-1)).mean()

    def sampling(self, mean, log_var):
        epsilon = torch.randn(mean.shape[0], mean.shape[-1], device=mean.device)
        return mean + (log_var / 2).exp() * epsilon.unsqueeze(1)

    def forward(self, z):
        '''
        :param z: shape (b, latent_dim)
        '''
        z = F.leaky_relu(self.fc1(z))
        z = F.leaky_relu(self.fc2(z))
        z_mean = self.mean(z)
        z_log_var = self.log_var(z)
        kl_loss = self.kl_loss(z_mean, z_log_var)
        enc_z = self.sampling(z_mean, z_log_var)
        if not self.training:
            enc_z = z_mean
        return enc_z, kl_loss

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, bottle_dim=20) -> None:
        super().__init__()
        self.fc1 = nn.Linear(bottle_dim, latent_dim//4)
        self.fc2 = nn.Linear(latent_dim//4, latent_dim//2)
        self.fc3 = nn.Linear(latent_dim//2, latent_dim)

    def forward(self, enc_z):
        z = F.leaky_relu(self.fc1(enc_z))
        z = F.leaky_relu(self.fc2(z))
        z = self.fc3(z)
        return z

class PluginVAE(nn.Module):
    def __init__(self, latent_dim=128, bottle_dim=20, kl_weight=1.0, beta=5.0) -> None:
        super().__init__()
        self.kl_weight = kl_weight
        self.beta = beta

        self.encoder = Encoder(latent_dim, bottle_dim)
        self.decoder = Decoder(latent_dim, bottle_dim)

    def set_beta(self, beta):
        self.beta = beta

    def forward(self, z):
        enc_z, kl_loss = self.encoder(z)
        z_out = self.decoder(enc_z)
        return z_out, kl_loss

    def loss(self, z):
        z_out, kl_loss = self.forward(z)
        z_loss = ((z_out-z)**2).mean()
        loss = z_loss + self.kl_weight * (kl_loss - self.beta).abs()
        return loss, kl_loss