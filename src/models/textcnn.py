"""TextCNN (Kim, 2014) with frozen pretrained embeddings + word vocabulary helpers."""
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_RE = re.compile(r"[a-z0-9']+|[!?.,]")
PAD, UNK = 0, 1


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts, max_size=50000, min_freq=2):
    cnt = Counter(t for s in texts for t in tokenize(s))
    words = [w for w, c in cnt.most_common(max_size) if c >= min_freq]
    return {"<pad>": PAD, "<unk>": UNK, **{w: i + 2 for i, w in enumerate(words)}}


def encode(text, vocab, max_len=256):
    ids = [vocab.get(t, UNK) for t in tokenize(text)][:max_len]
    return ids + [PAD] * (max_len - len(ids))


def load_glove(path, vocab, dim=300, seed=0):
    """Embedding matrix from a GloVe txt file; words not in GloVe get small random vectors."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(0, 0.1, (len(vocab), dim)).astype("float32")
    mat[PAD] = 0
    found = 0
    with open(path, encoding="utf8") as f:
        for line in f:
            w, *vec = line.rstrip().split(" ")
            if w in vocab and len(vec) == dim:
                mat[vocab[w]] = np.asarray(vec, dtype="float32")
                found += 1
    return torch.from_numpy(mat), found


class TextCNN(nn.Module):
    def __init__(self, embeddings: torch.Tensor, num_classes=5, filter_sizes=(3, 4, 5),
                 num_filters=100, dropout=0.5, freeze_embeddings=True):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=freeze_embeddings, padding_idx=PAD)
        d = embeddings.shape[1]
        self.convs = nn.ModuleList([nn.Conv1d(d, num_filters, k) for k in filter_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, ids):
        x = self.emb(ids).transpose(1, 2)               # (B, D, L)
        feats = [F.relu(c(x)).max(dim=2).values for c in self.convs]  # max-over-time pooling
        return self.fc(self.drop(torch.cat(feats, dim=1)))
