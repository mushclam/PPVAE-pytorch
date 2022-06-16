import argparse
from ctypes import ArgumentError
from pathlib import Path

import torch
from model.dataset import Vocab
from model.pluginVAE import PluginVAE
from utils import eval
import json

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except:
    from yaml import Loader, Dumper
from easydict import EasyDict

from model.pretrainVAE import PretrainVAE

def generate(args):
    # Get input data
    if hasattr(args, 'input'):
        input_sentence = args.input
    else:
        input_sentence = None

    # Load vocab
    vocab = Vocab.load(args.vocab_path, args.token)

    # Setup model
    if not hasattr(args, 'pretrain'):
        raise ArgumentError('PretrainVAE are MUST loaded to generate.')
    pretrain = PretrainVAE(max_vocab=len(vocab)).to(args.device)
    pretrain_ckp = torch.load(args.pretrain.path, map_location=args.device)
    pretrain.load_state_dict(pretrain_ckp['model'])
    pretrain.eval()

    if not hasattr(args, 'plugin'):
        args.plugin = None
    if args.plugin is not None:
        plugin = PluginVAE().to(args.device)
        plugin_ckp = torch.load(args.plugin.path, map_location=args.device)
        plugin.load_state_dict(plugin_ckp['model'])
        plugin.eval()
    else:
        plugin = None
    # Turn on model eval

    # Generation
    if input_sentence is not None:
        output_sentence = eval.gen_from_sentence(
            input_sentence, vocab, pretrain, None
        )
        output_sentence = output_sentence[1:-1]
        print(output_sentence)
        print(' '.join(output_sentence))
    else:
        gen_num = 5
        gen = eval.gen_from_ae(
            1.0, gen_num, vocab, pretrain, plugin,
            latent_dim=args.pretrain.latent_dim,
            max_len=args.max_len,
            argmax_flag=True,
            device=args.device
        )
        for g in gen[0]:
            print(' '.join([vocab.id2char[str(idx)] for idx in g]))


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