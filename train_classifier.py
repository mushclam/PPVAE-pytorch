import os
import sys
import json
import argparse
import pickle
from tqdm import tqdm

from yaml import load, dump

from classifier import Classifier
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except:
    from yaml import Loader, Dumper
from easydict import EasyDict

import torch
from torch.utils.data import DataLoader

from dataset import Vocab, ClassifierDataset

# To train the classifier to determine
# whether the sentences are positive or negative.

def load_pickle(path):
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result

def model_train(model, optimizer, loader, epoch, device, batch_size=256):
    model.train()
    bs_num = len(loader)
    total_loss = 0
    total_acc = 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        loss, pred = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.where(pred > 0.5, 1, 0)
        total_loss += loss.item()
        total_acc += (pred == y).sum().item() / batch_size

    print(f'epoch: {epoch}, loss: {total_loss/bs_num}, acc: {total_acc/bs_num}')

@torch.no_grad()
def model_eval(model, loader, device, batch_size=256):
    model.eval()
    bs_num = len(loader)
    total_loss = 0
    total_acc = 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        loss, pred = model(x, y)

        total_loss += loss.item()
        total_acc += (pred == y).sum().item() / batch_size
    return total_loss/bs_num, total_acc/bs_num

def main(args):
    train_decl = load_pickle(args.decl.train)
    train_poem = load_pickle(args.poem.train)
    valid_decl = load_pickle(args.decl.valid)
    valid_poem = load_pickle(args.poem.valid)
    test_decl = load_pickle(args.decl.test)
    test_poem = load_pickle(args.poem.test)

    # To balance the data ratio
    train_poem_size = len(train_poem)
    valid_poem_size = len(valid_poem)
    test_poem_size = len(test_poem)
    train_decl = train_decl[:train_poem_size]
    valid_decl = valid_decl[:valid_poem_size]
    test_decl = valid_decl[:test_poem_size]
    # Labeling
    train_decl = [[d, 0] for d in train_decl]
    valid_decl = [[d, 0] for d in valid_decl]
    test_decl = [[d, 0] for d in test_decl]
    train_poem = [[d, 1] for d in train_poem]
    valid_poem = [[d, 1] for d in valid_poem]
    test_poem = [[d, 1] for d in test_poem]

    train = train_decl + train_poem
    valid = valid_decl + valid_poem
    test = test_decl + test_poem

    # load vocab file
    with open(args.vocab_path, 'r') as f:
        chars, id2char, char2id = json.load(f)
        vocab = Vocab(
            chars, id2char, char2id,
            args.token.oov_token,
            args.token.start_token,
            args.token.end_token,
            args.token.pad_token
        )
    print('vocab size:', len(vocab))

    train_dataset = ClassifierDataset(train, vocab, max_len=args.max_len)
    valid_dataset = ClassifierDataset(valid, vocab, max_len=args.max_len)
    test_dataset = ClassifierDataset(test, vocab, max_len=args.max_len)

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

    model = Classifier(
        max_vocab=len(vocab),
        emb_size=args.emb_size,
        filter=args.filter,
        kernel=args.kernel
    ).to(args.device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(args.mu, args.nu))

    total_epoch = 100
    best_val = 0.0
    best_result = []

    for epoch in range(total_epoch):
        model_train(model, optimizer, train_dataloader, epoch, args.device, batch_size=args.batch_size)
                
        val_loss, val_acc = model_eval(model, valid_dataloader, args.device)
        print(f'val loss: {val_loss}, val acc: {val_acc}')
        if val_acc >= best_val:
            best_val = val_acc
            print('saving weights with best val:', val_loss)
            torch.save({
                'epoch': epoch,
                'loss': val_loss,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(args.output_dir, args.model_path))

    test_loss, test_acc = model_eval(model, test_dataloader, args.device)
    print(f'test loss: {test_loss}, test acc: {test_acc}')


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