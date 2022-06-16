import os
import sys
import argparse
import pickle
import json

from tqdm import tqdm
from yaml import load, dump
from model.classifier import Classifier

from model.pluginVAE import PluginVAE
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
import numpy as np

from utils import train, utils, eval


@torch.no_grad()
def evaluation(loader, pretrain, plugin, device):
    plugin.eval()
    total_loss = 0
    total_kl_loss = 0
    bs_num = len(loader)
    for x, _, _ in tqdm(loader, desc='Valid'):
        x = x.to(device)
        z, _ = pretrain.encoder(x)
        loss, kl_loss = plugin.loss(z)

        total_loss += loss.item()
        total_kl_loss += kl_loss.item()

    return total_loss/bs_num, total_kl_loss/bs_num

def padding(x, max_len):
    return torch.tensor([s + [0]*(max_len-len(s)) for s in x])

def main(args):
    train = utils.load_pickle(args.train_path)
    valid = utils.load_pickle(args.valid_path)
    print('train corpus size:', sum([len(d) for d in train]))
    print('sequences:', len(train))

    vocab = Vocab.load(args.vocab_path, args.token)
    print('vocab size:', len(vocab))
    # Dataset setup
    train_dataset = CustomDataset(train, vocab, max_len=args.max_len)
    valid_dataset = CustomDataset(valid, vocab, max_len=args.max_len)
    # Dataloader setup
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size
    )

    train_bs_num = len(train_dataloader)
    valid_bs_num = len(valid_dataloader)

    # Model setup
    pretrainVAE = PretrainVAE(
        max_vocab=len(vocab),
        emb_size=args.pretrain.emb_size,
        latent_dim=args.pretrain.latent_dim,
        nambda=args.pretrain.nambda
    ).to(args.device)
    pluginVAE = PluginVAE(
        latent_dim=args.plugin.latent_dim,
        bottle_dim=args.plugin.bottle_dim,
        kl_weight=args.plugin.kl_weight,
        beta=args.plugin.beta
    ).to(args.device)
    classifier = Classifier(
        max_vocab=len(vocab),
        emb_size=args.classifier.emb_size,
        filter=args.classifier.filter,
        kernel=args.classifier.kernel
    ).to(args.device)

    pretrain_checkpoint = torch.load(args.pretrain.path)
    pretrainVAE.load_state_dict(pretrain_checkpoint['model'])
    pretrainVAE.eval()

    classifier_checkpoint = torch.load(args.classifier.path)
    classifier.load_state_dict(classifier_checkpoint['model'])
    classifier.eval()

    # Optimizer
    optimizer = torch.optim.Adam(
        params=pluginVAE.parameters(),
        lr=args.lr, betas=(args.mu, args.nu)
    )
    # lr = 3e-4

    # training process
    end_iter = 100000
    total_epoch = end_iter // train_bs_num
    best_val = 100000.0
    best_result = []

    # to set weight beta, please refer to our paper
    def get_beta_weight(iter_num):
        now_beta_weight = min((5.0/10000)*iter_num, 5.0)
        return now_beta_weight

    total_iter = 0
    total_loss = 0
    total_kl_loss = 0
    for epoch in range(total_epoch):
        for i, (x, _, _) in enumerate(tqdm(train_dataloader)):
            x = x.to(args.device)
            with torch.no_grad():
                z, _ = pretrainVAE.encoder(x)
            pluginVAE.set_beta(get_beta_weight(total_iter))
            loss, kl_loss = pluginVAE.loss(z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
                
            total_iter += 1
        print (
            f'epoch: {epoch}, ', 
            f'avg_loss: {total_loss/train_bs_num}, ',
            f'avg_kl_loss: {total_kl_loss/train_bs_num}'
        )
       
        gen_num = 1000
        gen_samples1, gen_samples2, gen_samples3 = eval.gen_from_ae(
            1.0, gen_num, vocab, pretrainVAE, pluginVAE, args.plugin.bottle_dim, args.max_len, True, args.device)
        eval.get_distinct(gen_samples1)
        gen_samples1, gen_samples2, gen_samples3 = \
            [padding(s, max_len=args.max_len) for s in [gen_samples1, gen_samples2, gen_samples3]]
        gen_samples1 = gen_samples1.to(args.device)
        gen_result = classifier(gen_samples1, None)
        print('%f of the sample is positive in generator'%(1-gen_result.round().sum()/gen_num))
        # eval.gen_diversity(10, 0.5,)

        val_loss, val_kl_loss = evaluation(valid_dataloader, pretrainVAE, pluginVAE, args.device)
        print(f'valid loss: {val_loss}, kl loss: {val_kl_loss}')
        
        if val_loss <= best_val:
            best_val = val_loss
            best_result = val_loss
            print('saving weights with best val:', val_loss)
            torch.save({
                'epoch': epoch,
                'loss': best_val,
                'model': pluginVAE.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f'checkpoints/{args.log_tag}_best.pkl')
        else:
            torch.save({
                'epoch': epoch,
                'loss': best_val,
                'model': pluginVAE.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f'checkpoints/{args.log_tag}_last.pkl')

    # generating 10K text for evaluation
    gen_num = 10000
    gen_1, gen_2, gen_3 = eval.gen_from_ae(
            1.0, gen_num, vocab, pretrainVAE, pluginVAE, args.plugin.bottle_dim, args.max_len, True, args.device)
    with open('gen/PPVAE-single.txt', 'w', encoding='utf-8') as f:
        for g in gen_1:
            f.write(' '.join([vocab.id2char[str(index)] for index in g])+'\n')

    # get distinct-1/2
    eval.get_distinct(gen_1)

    # condition accurarcy by pre-trained classifier
    # notice length condition doesn't need classifier
    gen_1, gen_2, gen_3 = \
            [padding(s, max_len=args.max_len) for s in [gen_1, gen_2, gen_3]]
    gen_1 = gen_1.to(args.device)
    cls_result = classifier(gen_1, None)
    print('%f of the sample is positive'%(1-cls_result.round().sum()/gen_num))

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