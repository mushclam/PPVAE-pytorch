import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from dataset import ClassifierDataset, Vocab
from utils import eval
import json
from tqdm import tqdm

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except:
    from yaml import Loader, Dumper
from easydict import EasyDict

from classifier import Classifier
from utils import utils

def model_load(args):
    # Load vocab
    args.vocab_path = Path(args.vocab_path)
    with args.vocab_path.open('r') as f:
        chars, id2char, char2id = json.load(f)
        vocab = Vocab(
            chars, id2char, char2id,
            args.token.oov_token,
            args.token.start_token,
            args.token.end_token,
            args.token.pad_token
        )

    # Setup model
    classifier = Classifier(
        max_vocab=len(vocab),
        emb_size=args.emb_size
    ).to(args.device)

    # Load model (PretrainVAE)
    checkpoint = torch.load(args.model_path)
    classifier.load_state_dict(checkpoint['model'])

    return vocab, classifier

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

def generate(args):
    # model load
    vocab, classifier = model_load(args)

    # Get input data
    if args.input:
        data = args.input
        data = eval.preprocess(
        data, vocab, max_len=args.max_len).to(args.device)

        classifier.eval()
        predicted_class = classifier(data, None)
        predicted_class = torch.where(predicted_class > 0.5, 1, 0)
        predicted_class = 'Poem' if predicted_class.item() else 'Declarative'
        print(predicted_class)
        return
    else:
        test_decl = utils.load_pickle(args.decl.test)
        test_poem = utils.load_pickle(args.poem.test)

        # To balance the data ratio
        test_poem_size = len(test_poem)
        test_decl = test_decl[:test_poem_size]
        # Labeling
        test_decl = [[d, 0] for d in test_decl]
        test_poem = [[d, 1] for d in test_poem]
        test = test_decl + test_poem

        dataset = ClassifierDataset(test, vocab, max_len=args.max_len)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)

        classifier.eval()
        loss, acc = model_eval(classifier, dataloader, args.device, batch_size=args.batch_size)
        print(f'test loss: {loss}, test acc: {acc}')
        return

if __name__=='__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        args_y = load(f, Loader=Loader)
        args_y = EasyDict(args_y)
    print(args_y)

    # Device setup
    if args_y.device != 'cpu':
        args_y.device = f'cuda:{args_y.device}' if torch.cuda.is_available() else 'cpu'

    generate(args_y)