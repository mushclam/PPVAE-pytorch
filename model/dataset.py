from pathlib import Path
import torch
from torch.utils.data import Dataset
import json

# Vocab [chars, id2char, char2id]
class Vocab:
    def __init__(self, chars, id2char, char2id,
        oov_token, start_token, end_token, pad_token
    ) -> None:
        self.chars = chars
        self.id2char = id2char
        self.char2id = char2id
        self.oov_token = oov_token
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token

    def str2id(self, s, start_end = False):
        ids = [self.char2id.get(c, self.oov_token) for c in s]
        if start_end:
            ids = [self.start_token] + ids + [self.end_token]
        return ids

    def id2str(self, ids):
        return [self.id2char[str(x)] for x in ids]

    def __len__(self):
        # add special num of special tokens
        return len(self.chars) + 4

    @classmethod
    def load(cls, path, token):
        if not isinstance(path, Path):
            path = Path(path)
        with path.open('r') as f:
            chars, id2char, char2id = json.load(f)
            
            return cls(
                chars, id2char, char2id,
                token.oov_token,
                token.start_token,
                token.end_token,
                token.pad_token
            )

class CustomDataset(Dataset):
    def __init__(self, data, vocab: Vocab, max_len=17) -> None:
        super().__init__()
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get data
        d = self.data[index]
        # Crop sentence
        if len(d) > (self.max_len-2):
            d = d[:self.max_len-2]
        # Tokenize
        d = self.vocab.str2id(d, start_end=True)
        # Pad to tensor
        x = self.padding(d, self.max_len)
        y = self.padding(d, self.max_len)
        z = self.padding(d[1:], self.max_len)
        return x, y, z

    def padding(self, x, max_len):
        return torch.tensor(x + [0] * (max_len-len(x)))

class ClassifierDataset(CustomDataset):
    def __getitem__(self, index):
        words, label = self.data[index]

        if len(words) > (self.max_len-2):
            words = words[:self.max_len-2]
        words = self.vocab.str2id(words)
        words = self.padding(words, self.max_len)

        return words, label