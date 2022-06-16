import os
import json
import pickle
import argparse

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except:
    from yaml import Loader, Dumper
from easydict import EasyDict

def read_file(path):
    result = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower().split(' ') 
            result.append(line)
    return result

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

def main(args):
    train = read_file(args.train_path)
    valid = read_file(args.valid_path)
    test = read_file(args.test_path)

    print('train corpus size:', sum([len(d) for d in train]))
    print('sequences:', len(train))

    save_pickle(train, os.path.join(args.output_dir, 'train.pkl'))
    save_pickle(valid, os.path.join(args.output_dir, 'valid.pkl'))
    save_pickle(test, os.path.join(args.output_dir, 'test.pkl'))

    vocab = make_vocab(
        train,
        args.max_vocab,
        args.token.start_token,
        args.token.end_token,
        args.token.oov_token,
        args.token.pad_token
    )
    with open(os.path.join(args.output_dir, 'vocab.json'), 'w') as f:
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