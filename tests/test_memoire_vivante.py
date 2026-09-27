"""
Tests de la memoire vivante. Lancer : python tests/test_memoire_vivante.py
(ou pytest). CPU, quelques secondes.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import TransformerConfig
from src.memorizing_transformer import MemorizingTransformer
from src.knn_memory import KNNMemory

T = 16
CFG = TransformerConfig(vocab_size=50, n_layers=3, n_heads=2, d_model=32, d_ff=64,
                        max_seq_len=T, dropout=0.0)


def make_model(seed=0):
    torch.manual_seed(seed)
    m = MemorizingTransformer(CFG, mem_layer=1, topk=4)
    m.eval()
    return m


def run(model, segments, vram_slots=8, mem_size=64):
    """Lit les segments dans l'ordre avec une cave neuve. Retourne les logits de chacun."""
    B = segments[0].shape[0]
    state = model.new_state(B, mem_size=mem_size, vram_slots=vram_slots)
    out = []
    with torch.no_grad():
        for i, seg in enumerate(segments):
            reset = torch.full((B,), i == 0)
            out.append(model(seg, state=state, reset=reset)[0])
    return out, state


def rand_seg(B=2, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randint(0, CFG.vocab_size, (B, T), generator=g)


def test_premier_segment_sans_cave():
    """Cave vide : le modele se comporte exactement comme sans memoire."""
    m = make_model()
    s1 = rand_seg(seed=1)
    with torch.no_grad():
        ref = m(s1)[0]
    (l1,), _ = run(m, [s1])
    assert torch.allclose(ref, l1, atol=1e-6)


def test_causal_dans_le_segment():
    """Changer les tokens apres t ne change aucune sortie <= t (pas de fuite du futur)."""
    m = make_model()
    s1, s2 = rand_seg(seed=1), rand_seg(seed=2)
    (_, a), _ = run(m, [s1, s2])
    t = 7
    s2b = s2.clone()
    s2b[:, t + 1:] = rand_seg(seed=3)[:, t + 1:]
    (_, b), _ = run(m, [s1, s2b])
    assert torch.allclose(a[:, :t + 1], b[:, :t + 1], atol=1e-6)
    assert not torch.allclose(a[:, t + 1:], b[:, t + 1:])


def test_la_cave_est_lue():
    """Le passe (segment 1) change les predictions du segment 2 via la cave."""
    m = make_model()
    s2 = rand_seg(seed=2)
    (_, a), _ = run(m, [rand_seg(seed=1), s2])
    (_, b), _ = run(m, [rand_seg(seed=9), s2])
    assert not torch.allclose(a, b)


def test_cave_ne_contient_que_le_passe():
    """Apres un segment, la cave contient exactement T fiches par flux."""
    m = make_model()
    _, state = run(m, [rand_seg(seed=1)])
    assert state.memory.sizes().tolist() == [T, T]
    _, state = run(m, [rand_seg(seed=1), rand_seg(seed=2)])
    assert state.memory.sizes().tolist() == [2 * T, 2 * T]


def test_reset_par_flux():
    """Un changement de document ne vide que le flux concerne."""
    m = make_model()
    state = m.new_state(2, mem_size=64, vram_slots=8)
    with torch.no_grad():
        m(rand_seg(seed=1), state=state, reset=torch.tensor([True, True]))
        m(rand_seg(seed=2), state=state, reset=torch.tensor([True, False]))
    assert state.memory.sizes().tolist() == [T, 2 * T]


def test_le_caviste_ne_change_pas_le_calcul():
    """Le caviste decide seulement ce qui serait en VRAM : sorties identiques."""
    m = make_model()
    segs = [rand_seg(seed=i) for i in range(4)]
    a, _ = run(m, segs, vram_slots=2)
    b, _ = run(m, segs, vram_slots=48)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-6)


def test_anneau():
    """Au-dela de max_size, les fiches les plus anciennes sont remplacees."""
    mem = KNNMemory(1, 1, 4, max_size=8)
    for i in range(3):
        k = torch.full((1, 1, 4, 4), float(i + 1))
        mem.write(k, k)
    assert mem.sizes().tolist() == [8]
    assert sorted(set(mem.values[0, 0, :, 0].tolist())) == [2.0, 3.0]


def test_gradient_vers_la_requete():
    """La requete recoit du gradient a travers les fiches recuperees."""
    torch.manual_seed(0)
    m = MemorizingTransformer(CFG, mem_layer=1, topk=4)
    m.train()
    state = m.new_state(2, mem_size=64, vram_slots=8)
    m(rand_seg(seed=1), state=state, reset=torch.tensor([True, True]))
    blk = m.blocks[1]
    _, loss = m(rand_seg(seed=2), rand_seg(seed=3), state=state, reset=torch.tensor([False, False]))
    loss.backward()
    assert blk.gate_logit.grad is not None and blk.gate_logit.grad.abs().sum() > 0
    assert blk.qkv.weight.grad.abs().sum() > 0


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passes")
