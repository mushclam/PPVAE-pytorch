import argparse
import os
import csv
import random
import math
import pickle
import json
from tqdm import tqdm
from pathlib import Path

import re
from bs4 import BeautifulSoup as BS
import hanja
from konlpy.tag import Kkma

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except:
    from yaml import Loader, Dumper
from easydict import EasyDict

from konlpy.tag import Mecab

def make_vocab(data, max_vocab, start_token, end_token, oov_token, pad_token):
    chars = {}
    for lyric in data:
        for w in lyric: 
            chars[w] = chars.get(w,0) + 1

    print('all vocab:', len(chars))

    sort_chars = sorted(chars.items(), key = lambda a:a[1], reverse=True)
    print(sort_chars[:10])
    chars = dict(sort_chars[:max_vocab])

    id2char = {i+4:j for i,j in enumerate(chars)}

    id2char[start_token] = '<BOS>'
    id2char[end_token] = '<EOS>'
    id2char[oov_token] = '<UNK>'
    id2char[pad_token] = '<PAD>'

    char2id = {j:i for i,j in id2char.items()}
    print('vocab size:', len(char2id))
    return [chars, id2char, char2id]

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def open_tsv(path, remove_header=True, remove_label=True):
    if isinstance(path, str):
        path = Path(path)

    with path.open('r') as f:
        tr = csv.reader(f, delimiter='\t')
        rows = [row for row in tr]
    if remove_header:
        rows = rows[1:]
    if remove_label:
        rows = [row[:2] for row in rows]
    return rows

def flatten(data):
    result = []
    for ds in data:
        result += [d for d in ds]
    return result

def morph_list(data, tagger, desc=None):
    result = []
    for s in tqdm(data, desc=desc):
        s = tagger.morphs(s)
        result.append(s)
    return result

def load_pickle(path):
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result

def main(args):
    train_data1 = load_pickle(args.data1.train)
    valid_data1 = load_pickle(args.data1.valid)
    test_data1 = load_pickle(args.data1.test)

    train_data2 = load_pickle(args.data2.train)
    valid_data2 = load_pickle(args.data2.valid)
    test_data2 = load_pickle(args.data2.test)

    train_data = train_data1 + train_data2
    valid_data = valid_data1 + valid_data2
    test_data = test_data1 + test_data2    

    output_path = Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir()

    save_pickle(train_data, os.path.join(args.output_dir, 'kornli_poem_train.pkl'))
    save_pickle(valid_data, os.path.join(args.output_dir, 'kornli_poem_valid.pkl'))
    save_pickle(test_data, os.path.join(args.output_dir, 'kornli_poem_test.pkl'))

    vocab = make_vocab(
        train_data,
        args.max_vocab,
        args.token.start_token,
        args.token.end_token,
        args.token.oov_token,
        args.token.pad_token
    )
    with open(os.path.join(args.output_dir, 'kornli_poem_vocab.json'), 'w') as f:
        json.dump(vocab, f)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        args_y = load(f, Loader=Loader)
        args_y = EasyDict(args_y)
    print(args_y)

    main(args_y)