import os
from random import random
import sys
import numpy as np
import torch
import torch.nn.functional as F

from model.dataset import Vocab

def sample(logits, diversity=1.0):
    # sample from te given prediction
    logits = logits.double() / diversity
    logits = F.softmax(logits, dim=-1)
    probas = torch.multinomial(logits, 1)
    return probas

def argmax(logits):
    logits = logits.double()
    logits = F.softmax(logits, dim=-1)
    return logits.argmax()

@torch.no_grad()
def gen_diversity(
        model,
        vocab: Vocab,
        latent_dim=128,
        max_len=17,
        argmax_flag=False,
        device=None,

    ):
    random_vec = torch.randn((1, max_len, latent_dim), device=device)
    for diversity in [0.5, 0.8, 1.0]:
        for _ in range(3):
            print('\n----- diversity:', diversity)
            index, word = gen_from_vec(
                diversity=diversity,
                vec=random_vec,
                vocab=vocab,
                model=model,
                argmax_flag=argmax_flag,
                max_len=max_len,
                device=device
            )
            print(' '.join(word))

@torch.no_grad()
def gen_bs(vec, model, vocab: Vocab, latent_dim=128, max_len=17, topk=3, device=None):
    """beam search
    """
    print('\nbeam search...')
    xid = torch.tensor([[vocab.start_token]] * topk, device=device)
    vec = vec.repeat(topk, 1).to(device)
    scores = [0] * topk
    for i in range(max_len): 
        x_seq = F.pad(xid, (0, max_len-len(xid[0]))).to(device)
        logits = model.decoder(x_seq, x_seq, vec)
        logits = logits[:, i, 3:]

        arg_topk = logits.argsort(dim=1)[:, -topk:] 
        _xid = [] 
        _scores = [] 
        if i == 0:
            for j in range(topk):
                _xid.append(xid[j].tolist() + [arg_topk[0, j].item()+3])
                _scores.append(scores[j] + logits[0, arg_topk[0, j]].item())
        else:
            for j in range(len(xid)):
                for k in range(topk): 
                    _xid.append(xid[j].tolist() + [arg_topk[j, k].item()+3])
                    _scores.append(scores[j] + logits[j, arg_topk[j, k]].item())
            _arg_topk = np.argsort(_scores)[-topk:] 
            _xid = [_xid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = []
        scores = []
        for k in range(len(xid)):
            yid.append(_xid[k])
            scores.append(_scores[k])
        xid = torch.tensor(yid)

    s = vocab.id2str(xid[np.argmax(scores)].tolist())
    print(' '.join(s))
    
@torch.no_grad()
def reconstruct(num, dataloader, model, vocab: Vocab, device=None):
    model.encoder.eval()
    loader = iter(dataloader)
    for _ in range(num):
        print('\nreconstructing, first false second true')
        s = next(loader)
        s = s[0][0]
        s_w =  ' '.join([vocab.id2char[str(x.item())] for x in s])
        print(s_w)
        s_v, _ = model.encoder(s.unsqueeze(0).to(device))
        s_r, s_v1 = gen_from_vec(0.8, s_v, vocab, model, False, device=device)
        s_r, s_v2= gen_from_vec(0.8, s_v, vocab, model, True, device=device)
        print(' '.join(s_v1))
        print(' '.join(s_v2))
        
@torch.no_grad()
def gen_from_vec(diversity, vec, vocab: Vocab, model, argmax_flag, max_len=17, device=None):
    start_index = vocab.start_token #<BOS>
    start_word = vocab.id2char[str(start_index)]

    generated = [[start_index]]
    word = [[start_word]]

    while(vocab.end_token not in generated[0] and len(generated[0]) <= max_len):
        x_seq = F.pad(
            torch.tensor(generated, device=device),
            (0, max_len-len(generated[0]))
        )
        logits = model.decoder(x_seq, x_seq, vec)
        logits = logits[0, len(generated[0])-1, 3:]
        if argmax_flag:
            next_index = argmax(logits)
        else:
            next_index = sample(logits, diversity)
        next_index += 3
        next_word = vocab.id2char[str(next_index.item())]

        generated[0] += [next_index]
        word[0] += [next_word]
    return generated, word[0]

@torch.no_grad()
def gen_from_ae(diversity, num, vocab, pretrain, plugin,
    bottle_dim=20, latent_dim=128, max_len=17, argmax_flag=False, device=None
):
    r1, r2, r3 = [], [], []

    for _ in range(num):
        if plugin is not None:
            random_vec = torch.randn((1, max_len, bottle_dim), device=device)
            g_vec = plugin.decoder(random_vec)
        else:
            g_vec = torch.randn((1, max_len, latent_dim), device=device)

        generated, gen_word = gen_from_vec(
            diversity, g_vec, vocab, pretrain, argmax_flag, max_len, device
        )

        if '<EOS>' == gen_word[-1]:
            gen_word = gen_word[:-1]
        gen_word = gen_word[:(max_len-2)]    
        r1.append([vocab.char2id[c] for c in gen_word])
        r2.append([vocab.start_token]+[vocab.char2id[c] for c in gen_word]+[vocab.end_token])
        r3.append([vocab.char2id[c] for c in gen_word]+[vocab.end_token])
    return r1, r2, r3

def preprocess(sentence, vocab, max_len=17):
    def padding(x, max_len):
        return torch.tensor(x + [0] * (max_len-len(x)))

    if len(sentence) > (max_len-2):
        sentence = sentence[:max_len-2]
    sentence = vocab.str2id(sentence, start_end=True)
    sentence = padding(sentence, max_len).unsqueeze(0)
    return sentence

@torch.no_grad()
def gen_from_sentence(
    sentence, vocab, pretrain, plugin,
    diversity=0.8, max_len=17, argmax_flag=False, device=None
):
    if isinstance(sentence, str):
        sentence = preprocess(sentence, vocab, max_len)
    vec, _ = pretrain.encoder(sentence)
    if plugin is not None:
        vec = plugin(vec)

    generated, gen_word = gen_from_vec(
        diversity, vec, vocab, pretrain, argmax_flag, max_len, device
    )
    
    return gen_word

def get_distinct(id_list_data):
    grams = id_list_data
    grams_list1 = []
    for sen in grams:
        for g in sen:
            grams_list1.append(g)
            
    grams_list2 = []
    for sen in grams:
        for i in range(len(sen)-1):
            grams_list2.append(str(sen[i])+' '+str(sen[i+1]))
            
    print('distinct-1:', len(set(grams_list1))/len(grams_list1))
    print('distinct-2:', len(set(grams_list2))/len(grams_list2))

@torch.no_grad()
def interpolate(diversity, num, dataloader, model, vocab: Vocab, device=None):
    model.encoder.eval()
    loader = iter(dataloader)
    s1 = next(loader)[0][0]
    s2 = next(loader)[0][0]
    
    vec1, _ = model.encoder(s1.unsqueeze(0).to(device))
    vec2, _ = model.encoder(s2.unsqueeze(0).to(device))
    print('interpolate with sampling')
    print(' '.join([vocab.id2char[str(x.item())] for x in s1]))
    # sys.stdout.flush()
    for i in range(1, num+1):
        alpha = i/(num+1)
        vec = (1-alpha) * vec1 + alpha * vec2
        idx, w = gen_from_vec(diversity, vec, vocab, model, argmax_flag=False, device=device)
        print(' '.join(w))
    print(' '.join([vocab.id2char[str(x.item())] for x in s2]))
    # sys.stdout.flush()
    
    print('interpolate with argmax')
    print(' '.join([vocab.id2char[str(x.item())] for x in s1]))
    # sys.stdout.flush()
    for i in range(1, num+1):
        alpha = i/(num+1)
        vec = (1-alpha) * vec1 + alpha * vec2
        idx, w = gen_from_vec(diversity, vec, vocab, model, argmax_flag=True, device=device)
        print(' '.join(w))
    print(' '.join([vocab.id2char[str(x.item())] for x in s2]))
    # sys.stdout.flush()

def calculate_gradient_penalty(real_data, fake_data, real_outputs, fake_outputs, k=2, p=6, device=torch.device("cpu")):
    real_grad_outputs = torch.full([*real_outputs.shape], 1, dtype=torch.float32, requires_grad=False, device=device)
    fake_grad_outputs = torch.full([*fake_outputs.shape], 1, dtype=torch.float32, requires_grad=False, device=device)

    real_gradient = torch.autograd.grad(
        outputs=real_outputs,
        inputs=real_data,
        grad_outputs=real_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    fake_gradient = torch.autograd.grad(
        outputs=fake_outputs,
        inputs=fake_data,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Edit cal for each word
    real_gradient_norm = real_gradient.pow(2).sum(-1) ** (p / 2)
    fake_gradient_norm = fake_gradient.pow(2).sum(-1) ** (p / 2)

    gradient_penalty = torch.mean(real_gradient_norm + fake_gradient_norm) * k / 2
    return gradient_penalty