"""Chargement et preparation des donnees pour Nature."""

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from typing import List, Tuple

from src.config import DataConfig


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def is_clean_line(line: str) -> bool:
    """Filtre les lignes corrompues ou de mauvaise qualite."""
    line = line.strip()
    if len(line) < 10:
        return False
    # Ratio de caracteres alphanumeriques + espaces + ponctuation courante
    alnum_count = sum(1 for c in line if c.isalnum() or c.isspace() or c in '.,;:!?\'"-()')
    ratio = alnum_count / len(line) if len(line) > 0 else 0
    if ratio < 0.7:
        return False
    # Trop de majuscules consecutives (headers Gutenberg etc.)
    if re.search(r'[A-Z]{20,}', line):
        return False
    # Lignes qui sont juste des separateurs
    if re.match(r'^[\s\-=*_#]+$', line):
        return False
    return True


def load_texts(config: DataConfig) -> str:
    """Charge et concatene tous les fichiers texte filtres."""
    all_text = []
    files_used = []
    files_skipped = []

    for fname in sorted(os.listdir(config.data_dir)):
        if not fname.endswith('.txt'):
            continue
        if fname in config.exclude_files:
            files_skipped.append(fname)
            continue
        if fname.endswith('.json'):
            continue

        fpath = os.path.join(config.data_dir, fname)
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        clean_lines = [l.strip() for l in lines if is_clean_line(l)]
        if clean_lines:
            all_text.append('\n'.join(clean_lines))
            files_used.append(f"{fname} ({len(clean_lines)} lignes)")

    print(f"Fichiers charges: {len(files_used)}")
    for f in files_used:
        print(f"  {f}")
    if files_skipped:
        print(f"Fichiers exclus: {', '.join(files_skipped)}")

    return '\n'.join(all_text)


def tokenize_text(text: str, tokenizer: Tokenizer) -> List[int]:
    """Tokenize le texte complet."""
    encoded = tokenizer.encode(text)
    return encoded.ids


class TextDataset(Dataset):
    """Dataset de sequences pour language modeling causal."""

    def __init__(self, token_ids: List[int], seq_len: int):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        # Nombre de sequences completes disponibles
        self.n_sequences = (len(self.token_ids) - 1) // seq_len

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.token_ids[start:start + self.seq_len]
        y = self.token_ids[start + 1:start + self.seq_len + 1]
        return x, y


def prepare_data(config: DataConfig) -> Tuple[TextDataset, TextDataset, Tokenizer]:
    """Prepare les datasets train et val."""
    tokenizer = load_tokenizer(config.tokenizer_path)
    text = load_texts(config)

    print(f"\nTexte total: {len(text):,} caracteres")
    token_ids = tokenize_text(text, tokenizer)
    print(f"Tokens totaux: {len(token_ids):,}")

    # Split train/val
    n_val = int(len(token_ids) * config.val_ratio)
    val_ids = token_ids[-n_val:]
    train_ids = token_ids[:-n_val]

    train_dataset = TextDataset(train_ids, config.seq_len)
    val_dataset = TextDataset(val_ids, config.seq_len)

    print(f"Train: {len(train_dataset)} sequences de {config.seq_len} tokens")
    print(f"Val: {len(val_dataset)} sequences de {config.seq_len} tokens")

    return train_dataset, val_dataset, tokenizer
