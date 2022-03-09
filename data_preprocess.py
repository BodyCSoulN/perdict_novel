import collections
import re
import random

import torch
from torch.nn import functional as F
from torch import nn
from d2l import torch as d2l
import jieba

def read_novel(filepath='/data/datasets/d2l_data/cover_sky.txt'):
    with open(filepath, 'r', encoding='gbk') as f:
        lines = f.readlines()
        print('len of lines:', len(lines))
        # 简单的把空行、空格都去掉
        return [re.sub('\s', ' ', line) for line in lines]
    
def tokenize(lines, token='char', language='chinese'):
    # print('token = ', token)
    if token == 'word':
        if language == 'chinese':
            # print(jieba.lcut(lines[0], cut_all=False))
            return [jieba.lcut(line, cut_all=False) for line in lines]
        if language == 'english':
            return [line.split() for line in lines]
    if token == 'char':
        return [list(line) for line in lines]
    else:
        print('error token：', token)
        
def corpus_counter(tokens):
    if len(tokens) == 0 or isinstance(tokens, list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
            
        counter = corpus_counter(tokens)
        self.token_freq = sorted(counter.items(), key=lambda x:x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        self.idx_to_token, self.token_to_idx = [], dict()
        uniq_tokens += [
            token for token, freq in self.token_freq
            if freq > min_freq and token not in uniq_tokens
        ]
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
            
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (tuple, list)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_token(self, indices):
        if not isinstance(indices, (tuple, list)):
            return self.idx_to_token[indices]
        return [self.__getitem__(index) for index in indices]
    
def load_novel(token, language, max_tokens=-1):
    lines = read_novel()
    tokens = tokenize(lines, token, language)
    vocab = Vocab(tokens, 0)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab
    
def seq_data_iter_random(corpus, batch_size, time_steps):
    corpus = corpus[random.randint(0, time_steps - 1):]
    num_subseq = (len(corpus) - 1) // time_steps
    initial_indices = list(range(0, num_subseq * time_steps, time_steps))
    random.shuffle(initial_indices)
    
    def data(pos):
        return corpus[pos: pos + time_steps]
    
    num_batches = num_subseq // batch_size
    
    for i in range(0, num_batches * batch_size, batch_size):
        initial_indices_per_epoch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_epoch]
        Y = [data(j + 1) for j in initial_indices_per_epoch]
        yield torch.tensor(X), torch.tensor(Y)

class SeqSampler:
    def __init__(self, batch_size, time_steps, token, language, max_tokens=1000000):
        self.sampler = seq_data_iter_random
        self.batch_size = batch_size
        self.time_steps, self.max_tokens = time_steps, max_tokens
        self.corpus, self.vocab = load_novel(token, language, self.max_tokens)
    def __iter__(self):
        return self.sampler(self.corpus, self.batch_size, self.time_steps)
    
def load_data_novel(batch_size, time_steps, token, language, max_tokens):
    data_iter = SeqSampler(batch_size, time_steps, token, language, max_tokens)
    return data_iter, data_iter.vocab

def trans_dim(state):
    if isinstance(state, (tuple, list)):
        return [s.permute([1, 0, 2]) for s in state]
    else:
        return state.permute([1, 0, 2])