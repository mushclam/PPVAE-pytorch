from modules import PositionalEncoding, TiedEmbeddingsTransposed
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.eval import calculate_gradient_penalty

class Discriminator(nn.Module):
    def __init__(self, dim=128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        '''
        :param x: shape (batch, latent_dim)
        '''
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)

class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        max_vocab=10000,
        emb_size=256,
        gru_dim=150,
        embedding=None
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        if embedding is None:
            self.embedding = nn.Embedding(max_vocab, emb_size, padding_idx=0)
        else:
            self.embedding = embedding
        self.encoder = nn.GRU(emb_size, gru_dim, batch_first=True, bidirectional=True)
        self.mean = nn.Linear(2*gru_dim, self.latent_dim)
        self.log_var = nn.Linear(2*gru_dim, self.latent_dim)

    def kl_loss(self, mean, log_var):
        return (-0.5 * (1 + log_var - mean**2 - log_var.exp()).sum(-1)).mean()

    def sampling(self, mean, log_var):
        # epsilon = torch.randn(*mean.shape, device=mean.device)
        # return mean + (log_var / 2).exp() * epsilon
        # This is the answer
        epsilon = torch.randn(mean.shape[0], mean.shape[-1], device=mean.device)
        return mean + (log_var / 2).exp() * epsilon.unsqueeze(1)

    def forward(self, x):
        '''
        :param x: shape (b, max_len) type int32
        :return enc_z: shape (b, max_len, latent_dim) 
        '''
        x = self.embedding(x) # (b, max_len, emb_size)
        x = self.encoder(x)[0] # (b, max_len, 2 * gru_size)
        z_mean = self.mean(x) # (b, max_len, latent_dim)
        z_log_var = self.log_var(x) # (b, max_len, latent_dim)
        kl_loss = self.kl_loss(z_mean, z_log_var)

        enc_z = self.sampling(z_mean, z_log_var) # (b, max_len, latent_dim)
        if not self.training:
            enc_z = z_mean
        return enc_z, kl_loss


class DecoderInnerLayer(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        emb_size=256,
        num_heads=8
    ) -> None:
        super().__init__()
        head_size = (emb_size+latent_dim) // num_heads
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.attn = nn.MultiheadAttention(emb_size+latent_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(emb_size+latent_dim)
        self.fc2 = nn.Linear(emb_size+latent_dim, head_size*num_heads)
        self.layer_norm2 = nn.LayerNorm(head_size*num_heads)
        self.pos_enc = PositionalEncoding(head_size*num_heads)

    def forward(self, h, z):
        z_hier = self.fc1(z)
        h = torch.cat([h, z_hier], dim=-1)
        h_attn = self.attn(h, h, h)[0]
        h = h + h_attn
        h = self.layer_norm1(h)
        h_mlp = F.relu(self.fc2(h))
        h = h + h_mlp
        h = self.layer_norm2(h)
        h = self.pos_enc(h)
        return h

# Decoder is 3-layer transformer
# add extra positional embedding after each layer
class Decoder(nn.Module):
    def __init__(
        self,
        n_layers=3,
        emb_size=256,
        num_heads=8,
        latent_dim=128,
        max_vocab=10000,
        embedding=None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        # Sharing with encoder embedding layer
        if embedding is None:
            self.embedding = nn.Embedding(max_vocab, emb_size, padding_idx=0)
        else:
            self.embedding = embedding
        self.pos_enc = PositionalEncoding(emb_size)
        # Layers
        self.layers = nn.ModuleList([
            DecoderInnerLayer(latent_dim, emb_size+i*latent_dim, num_heads) \
                for i in range(n_layers)
        ])
        # Last FC layer
        self.fc = nn.Linear(emb_size+latent_dim*3, emb_size)
        # Embedded vector -> words
        # pytorch apply softmax to vector when calculate cross entropy loss
        # Therefore, we deactivate softmax argument in TiedEmbeddingsTransposed
        self.trans_emb = TiedEmbeddingsTransposed(tied_to=self.embedding)

    def forward(self, dec_in, dec_true, enc_z, softmax=False):
        '''
        :param x: shape (b, max_len)
        :param z: shape (b, latent_dim)
        :param true: shape(b, max_len)
        :return output: shape (b, )
        '''
        b, max_len = dec_in.shape
        dec_in = self.embedding(dec_in)
        decoder_h = self.pos_enc(dec_in) # shape (b, max_len, emb_size)
        if enc_z.dim() == 2:
            decoder_z = enc_z.unsqueeze(1).expand(b, max_len, self.latent_dim) # shape (b, max_len, max_len)
        elif enc_z.dim() == 3:
            decoder_z = enc_z

        for layer in self.layers:
            decoder_h = layer(decoder_h, decoder_z)

        decoder_h = self.fc(decoder_h)
        output = self.trans_emb(decoder_h) # (b, max_len, max_vocab)
        if softmax:
            output = F.softmax(output, dim=-1)
        return output

class PretrainVAE(nn.Module):
    def __init__(
        self,
        max_vocab=10000,
        emb_size=256,
        latent_dim=128,
        nambda=20
    ):
        super().__init__()
        self.nambda = nambda
        self.embedding = nn.Embedding(max_vocab, emb_size, padding_idx=0)
        self.encoder = Encoder(max_vocab=max_vocab, embedding=self.embedding)
        self.disc = Discriminator(latent_dim)
        self.decoder = Decoder(max_vocab=max_vocab, embedding=self.embedding)
        self.celoss = nn.CrossEntropyLoss(reduction='none')

    def masking(self, x):
        return (x.unsqueeze(2) > 0).double()

    def train_disc(self):
        self.disc.train()
        for p in self.disc.parameters():
            p.requires_grad = True

        # self.encoder.eval()
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False
    
    def train_ae(self):
        self.disc.eval()
        for p in self.disc.parameters():
            p.requires_grad = False

        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad = True
        
        self.decoder.train()
        for p in self.decoder.parameters():
            p.requires_grad = True


    def forward_disc(self, enc_in, z_in):
        '''
        :param enc_in: shape (b, max_len)
        :param z_in: shape (b, latent_dim)
        '''
        # fix encoder gradients
        if self.training:
            self.train_disc()

        b, max_len = enc_in.shape
        latent_dim = z_in.shape[-1]
        if z_in.dim() == 2:
            z_in = z_in.unsqueeze(1).expand(b, max_len, latent_dim)
        z_real_score = self.disc(z_in)

        # for gradient penalty
        z_fake, kl_loss = self.encoder(enc_in) # z_fake shape (b, max_len, latent)
        z_fake.requires_grad = True
        z_fake_score = self.disc(z_fake) # z_fake_score (b, max_len)

        # Loss
        d_loss = (z_real_score - z_fake_score).mean()
        # loss for gradient penalty
        grad_loss = calculate_gradient_penalty(z_in, z_fake, z_real_score, z_fake_score, device=z_in.device)
        # w_dist
        w_dist = (z_fake_score - z_real_score).mean()
        # total_loss
        total_loss = -(d_loss - grad_loss)

        return total_loss, (kl_loss, w_dist)

    def forward_ae(self, enc_in, dec_in, dec_true):
        '''
        :param enc_in: shape (b, max_len)
        :param dec_in: shape (b, max_len)
        :param dec_true: shape (b, max_len)
        '''
        if self.training:
            self.train_ae()

        # z fake score
        enc_z, kl_loss = self.encoder(enc_in)
        z_fake_score = self.disc(enc_z)
        # decoder input mask
        dec_true_mask = self.masking(dec_true)
        dec_out = self.decoder(dec_in, dec_true, enc_z)

        xent_loss = (self.celoss(dec_out.transpose(1, 2), dec_true)*dec_true_mask[:,:,0]).sum()\
            / (dec_true_mask[:,:,0]).sum()
        d_loss = (-z_fake_score).mean()
        total_loss = xent_loss + self.nambda*d_loss

        return total_loss, (kl_loss, xent_loss)

    def forward(self, *args, mode='train_disc'):
        if mode == 'train_disc':
            return self.forward_disc(*args)
        elif mode == 'train_ae':
            return self.forward_ae(*args)