"""Configuration pour le projet Nature."""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class DataConfig:
    data_dir: str = "D:/DaBrainRecurrent/data"
    tokenizer_path: str = "D:/DaBrainRecurrent/data/dabrain_bpe_8k.json"
    seq_len: int = 256
    # Fichiers a exclure (anglais, vides)
    exclude_files: List[str] = field(default_factory=lambda: [
        "corpus.txt", "corpus_test.txt", "candide.txt",
        "tasks_train3_part3.txt",  # vide
    ])
    val_ratio: float = 0.05  # 5% du train pour validation


@dataclass
class TransformerConfig:
    vocab_size: int = 8000
    n_layers: int = 6
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    weight_tying: bool = True


@dataclass
class MemoryConfig:
    n_elements: int = 2048
    n_neighbors: int = 32
    n_memory_layers: int = 3  # cross-attention apres chaque 2 blocs
    focus_k: int = 64         # taille du foyer : top-K elements actifs pour cross-attn
    alpha: float = 0.1        # taux de propagation
    delta: float = 0.05       # taux de decay
    gamma: float = 5.0        # taux focus -> activation
    epsilon: float = 1e-6
    recompute_graph_every: int = 100  # steps entre recalcul du graphe


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 5000
    val_every: int = 100
    log_every: int = 10
    save_every: int = 1000
    seed: int = 42
    device: str = "cuda"
    run_dir: str = "D:/Nature/runs"
    grad_clip: float = 1.0
