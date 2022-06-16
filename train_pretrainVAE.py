import os
import sys
import argparse
import pickle
import json

from tqdm import tqdm
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except:
    from yaml import Loader, Dumper
from easydict import EasyDict

import torch
from model.dataset import CustomDataset, Vocab
from model.pretrainVAE import PretrainVAE

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import eval
from utils import utils

def main(args):
    # Arguments setup
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    args.max_vocab += 4 # Add length of special tokens
    # Load data
    print('[INFO] Load datasets...', end='')
    train = utils.load_pickle(args.train_path)
    valid = utils.load_pickle(args.valid_path)
    test = utils.load_pickle(args.test_path)
    vocab = Vocab.load(args.vocab_path, args.token)
    print('Done.')
    # Dataset setup
    train_dataset = CustomDataset(train, vocab, max_len=args.max_len)
    valid_dataset = CustomDataset(valid, vocab, max_len=args.max_len)
    test_dataset = CustomDataset(test, vocab, max_len=args.max_len)
    # Dataloader setup
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    train_bs_num = len(train_dataloader)
    valid_bs_num = len(valid_dataloader)
    test_bs_num = len(test_dataloader)
    print(f'Number of batches: train {train_bs_num} valid {valid_bs_num} test {test_bs_num}')

    # Model setup
    model = PretrainVAE(
        max_vocab=args.max_vocab,
        emb_size=args.emb_size,
        latent_dim=args.latent_dim,
        nambda=args.nambda
    ).to(args.device)
    print(model)
    # Optimizer setup
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(args.mu, args.nu))

    end_iter = 1200000
    total_epoch = end_iter // train_bs_num
    best_val = 100000.0
    best_result = []

    metric = {
        'd_loss': 0,
        'g_loss': 0,
        'disc_kl_loss': 0,
        'total_w_dist': 0,
        'ae_kl_loss': 0,
        'total_ce_loss': 0
    }
    

    total_iter = 0
    writer = SummaryWriter(log_dir=f'runs/{args.log_tag}')
    for epoch in range(total_epoch):
        print(f'[INFO] Epoch {epoch}/{total_epoch}:')
        model.train()
        for i, (x, y, z) in enumerate(tqdm(train_dataloader)):
            x, y, z = [d.to(args.device) for d in (x, y, z)]
            if i % 4 != 3:
                # train discriminator for 3 steps
                b = x.shape[0]
                z_sample = torch.randn(
                    b, args.max_len, latent_dim, device=args.device, requires_grad=True)
                total_loss, (kl_loss, w_dist) = \
                    model(x, z_sample, mode='train_disc')
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                metric['d_loss'] += total_loss.item()
                metric['disc_kl_loss'] += kl_loss.item()
                metric['total_w_dist'] += w_dist.item()
            else:
                # train encoder-decoder for 1 step
                total_loss, (kl_loss, ce_loss) = \
                    model(x, y, z, mode='train_ae')
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                metric['g_loss'] += total_loss.item()
                metric['ae_kl_loss'] += kl_loss.item()
                metric['total_ce_loss'] += ce_loss.item()
            total_iter += 1

        split = train_bs_num / 4
        metric['d_loss'] = metric['d_loss']/(split*3)
        metric['g_loss'] = metric['g_loss']/split
        metric['disc_kl_loss'] = metric['disc_kl_loss']/(split*3)
        metric['total_w_dist'] = metric['total_w_dist']/(split*3)
        metric['ae_kl_loss'] = metric['ae_kl_loss']/split
        metric['total_ce_loss'] = metric['total_ce_loss']/split

        for k, v in metric.items():
            print(f'{k}: {v:4f}, ', end='')
            writer.add_scalar(f'{k}', v, total_iter)
            metric[k] = 0
        print()

        if epoch % 50 == 0:   
            eval.gen_diversity(model, vocab, latent_dim=args.latent_dim, max_len=args.max_len, device=args.device)
            eval.gen_diversity(model, vocab, latent_dim=args.latent_dim, max_len=args.max_len, device=args.device)
            eval.reconstruct(5, test_dataloader, model, vocab, device=args.device)
            for _ in range(5):
                vec =  torch.randn((1, latent_dim))
                eval.gen_bs(vec, model, vocab, topk=1, device=args.device)
            print('diversity 0.8')
            eval.interpolate(0.8, 8, test_dataloader, model, vocab, device=args.device)
            print('diversity 1.0')
            eval.interpolate(1.0, 8, test_dataloader, model, vocab, device=args.device)

        val_loss = evaluation(model, valid_dataloader, device=args.device)
        print(f'valid loss: {val_loss}')
        
        if val_loss <= best_val:
            best_val = val_loss
            best_result = val_loss
            print('saving weights with best val:', val_loss)
            torch.save({
                'epoch': epoch,
                'loss': best_val,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f'checkpoints/{args.log_tag}_best.pkl')
        else:
            torch.save({
                'epoch': epoch,
                'loss': best_val,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f'checkpoints/{args.log_tag}_last.pkl')

@torch.no_grad()
def evaluation(model, dataloader, device=None):
    model.eval()
    val_loss = 0
    for x, y, z in tqdm(dataloader):
        x, y, z = [d.to(device) for d in (x, y, z)]
        loss, (kl_loss, ce_loss) = model(x, y, z, mode='train_ae')
        val_loss += loss.item()
    return val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        args_y = load(f, Loader=Loader)
        args_y = EasyDict(args_y)
    print(args_y)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args_y.device = f'cuda:{args_y.device}'

    main(args_y)