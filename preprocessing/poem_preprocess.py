import argparse
import os
import csv
import random
import math
import pickle
import json
from tqdm import tqdm

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

def arrange(words):
    arrange_re = {
        '...': re.compile(r'…')
    }
    for trg, sub in arrange_re.items():
        words = sub.sub(trg, words)
    return words

def clean(words):
    clean_re = re.compile(r'[^ .,\\$?!~@%"\'|0-9|ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+')
    return clean_re.sub('', words)

def get_poem_from_file(filename):
    with open(filename, 'r') as f:
        soup = BS(f, 'lxml')
    title = soup.find('title').contents
    title = title[0] if len(title) != 0 else ''
    author = soup.find('author').contents
    author = author[0] if len(author) != 0 else ''
    date = soup.find('date').contents
    date = date[0] if len(date) != 0 else ''
    text = soup.find('tdmsfiletext')
    poem = []
    for p in text.find_all('poem'):
        if len(p.contents) == 0:
            continue
        p = p.contents[0]
        p = arrange(p).strip()
        # translate chinese char to korean char
        p = hanja.translate(p, 'substitution')
        # clean unused special chars
        p = clean(p)
        p = [title, author, date, p]
        poem.append(p)
    return poem

def main(args):
    poem = []
    for i in range(1, 100+1):
        filename = os.path.join(args.data_dir, f'{i:04d}.txt')
        p = get_poem_from_file(filename)
        poem += p

    # save poem unit data
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    poem_output = os.path.join(args.output_dir, 'arranged_poem.tsv')
    with open(poem_output, 'w') as f:
        f.write(f'title\tauthor\tdate\tpoem')
        for p in poem:
            f.write(f'{p[0]}\t{p[1]}\t{p[2]}\t{p[3]}\n')

    # filename = os.path.join(args.data_dir, 'arranged_poem.csv')
    # with open(filename, 'r', newline='') as f:
    #     rd = csv.reader(f, delimiter=';', quotechar='|')
    #     for row in rd:
    #         print(row)

    # poem split to sentence / too long to train
    for i, p in enumerate(poem):
        words = p[3]
        split = list(filter(None, words.split('$')))
        split = [list(filter(None, w.split('\\'))) for w in split]
        sents = []
        for s in split:
            if isinstance(s, list):
                sents += s
            else:
                sents.append(s)
        sents = [s.strip() for s in sents]
        poem[i][3] = sents

    # save sentence unit data
    sent_output = os.path.join(args.output_dir, 'arranged_sent.tsv')
    with open(sent_output, 'w') as f:
        f.write(f'title\tauthor\tdate\tsentence')
        for p in poem:
            title = p[0]
            author = p[1]
            date = p[2]
            for w in p[3]:
                f.write(f'{title}\t{author}\t{date}\t{w}\n')
    
    # Morph Analysis
    kkma = Kkma()
    ps = []
    for p in poem:
        ps += p[3]
    sentences = []
    for s in tqdm(ps):
        sentences.append(kkma.morphs(s))

    # Split data to train, valid, test
    random.shuffle(sentences)
    train_size = math.ceil(len(sentences)*0.8)
    valid_size = math.ceil(len(sentences)*0.1)
    test_size = len(sentences) - (train_size+valid_size)
    print(f'[Dataset size] train: {train_size}, valid: {valid_size}, test: {test_size}')

    train_data = sentences[0:train_size]
    valid_data = sentences[train_size:train_size+valid_size]
    test_data = sentences[train_size+valid_size:]

    # Save as pickle
    save_pickle(train_data, os.path.join(args.output_dir, 'poem_train.pkl'))
    save_pickle(valid_data, os.path.join(args.output_dir, 'poem_valid.pkl'))
    save_pickle(test_data, os.path.join(args.output_dir, 'poem_test.pkl'))

    # Make and Save vocabulary data
    vocab = make_vocab(
        train_data,
        args.max_vocab,
        args.token.start_token,
        args.token.end_token,
        args.token.oov_token,
        args.token.pad_token
    )
    with open(os.path.join(args.output_dir, 'poem_vocab.json'), 'w') as f:
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