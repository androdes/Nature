# Nature

*Not a transformer. A posture.*

Training a 1-billion-parameter transformer on a 12 GB GPU by spreading weights across VRAM, RAM, and disk. A demonstration that VRAM is a speed constraint, not a capacity constraint.

## The idea

The ML community treats VRAM as a hard limit on model size. If a model doesn't fit in VRAM, you need a bigger GPU or distributed training. But VRAM is just the fastest tier of a memory hierarchy. RAM is 10x larger and 10x slower. Disk is 100x larger and 100x slower. A system that uses all three tiers can train models far beyond what VRAM alone allows — it just takes longer.

This project demonstrates that tradeoff on consumer hardware. A 1B-parameter transformer trains on an RTX 3060 (12 GB VRAM), with its weights in VRAM, optimizer states shuttled through CPU RAM, and gradient checkpointing to limit activation memory. It converges correctly. It's slow (8 seconds per step instead of 0.5), but it works.

The tortoise beats the hare — not in speed, but in how far it can go.

## What was demonstrated

### 1. External memory consultation works

Adding a cross-attention layer between a transformer and a small external memory (512 learned vectors) improves validation loss by ~1.3% compared to a pure transformer with the same backbone. This is equivalent in quality to adding two extra transformer layers, but costs 5x fewer additional parameters (~330K vs 1.6M).

The surprising finding: the *content* of the memory doesn't matter. Learned parameters, token embeddings, FAISS-retrieved corpus chunks, and random vectors all produce the same ~1.3% gain. What helps is the structural pattern of consultation — a second information pathway alongside self-attention — not the information stored in it.

| Model | Parameters | Val Loss | vs Baseline |
|-------|-----------|----------|-------------|
| Baseline 6L/256d | 6.85M | 4.711 +/- 0.052 | reference |
| Baseline 8L/256d | 8.43M | 4.648 +/- 0.010 | -1.3% |
| SimpleMem (6L + 512 memory) | ~7.2M | 4.646 +/- 0.057 | -1.4% |
| Retrieval FAISS (K=64) | ~7.2M | 4.651 +/- 0.103 | -1.3% |

All results averaged over 3 random seeds. Training data: 2.6M tokens of diverse French text.

### 2. Hierarchical memory preserves quality

When the external memory lives partly in CPU RAM instead of entirely in VRAM, training quality is indistinguishable from the all-VRAM version. Measured gap: 0.1% (well within noise). The FAISS index handles retrieval from RAM transparently, and an LRU cache keeps frequently accessed elements close to the GPU.

### 3. A 1B model trains on a 12 GB GPU

Using gradient checkpointing and a block-by-block optimizer that keeps Adam states in CPU RAM, a 36-layer / 1536-dim transformer (1.03 billion parameters) trains on an RTX 3060. Peak VRAM usage: 8.3 GB for a model whose weights alone total 3.9 GB (the rest is activations and gradients). Optimizer states (7.8 GB for Adam momentum and variance) live entirely in CPU RAM and are loaded to VRAM one block at a time during the optimizer step.

The model converges — validation loss decreases steadily from 6.3 to 5.1 over 2000 steps. It doesn't reach the quality of the 6.8M baseline (4.71) because the training data is too small (2.6M tokens for 1B parameters), not because the mechanism fails.

| Model | Parameters | Peak VRAM | ms/step | Val Loss |
|-------|-----------|-----------|---------|----------|
| Baseline 6L | 6.8M | ~2 GB | ~200 | 4.711 |
| 350M all-VRAM | 353M | 6.9 GB | ~500 | 4.875 |
| 1B offloaded | 1,030M | 8.3 GB | ~8,000 | 5.116 |

### 4. The bottleneck is data, not architecture

At 2.6M training tokens, even a 6.8M-parameter model overfits (train loss 2.0 vs val loss 4.7). Larger models overfit faster. The external memory can't help because there's nothing new to retrieve — the transformer already memorizes the entire corpus in its weights. To see memory scaling pay off, you'd need 100M+ tokens.

## What didn't work

### The Activation Field Memory Transformer

The original architecture was bio-inspired: a "field" of activation values spread across memory elements, with a mobile focus that shifts through the field, propagating activation to semantic neighbors. Think of it as a simulation of how a thought "circulates" — when you think "cat", "mouse" and "garden" light up by association.

Ten experiments tested variants of this mechanism:
- Full attention over all memory elements (std of activation: 0.0007 — essentially zero)
- Top-K focus selection (std: 0.03, but static throughout training)
- Content-based focus routing
- Different initialization strategies and hyperparameters

None produced meaningful differentiation in the activation field. The mechanism was inert. The root cause: a bootstrap problem. Activations need focused attention to differentiate, but attention needs differentiated activations to focus. Both start uniform and stay uniform.

The ablation that revealed this: removing the activation field entirely and keeping just the cross-attention to memory (SimpleMem) produced *better* results than the full mechanism. The activation dynamics were not just unhelpful — they interfered with learning.

### Memory size scaling

Increasing the number of memory elements from 128 to 50,000 produces no improvement. At this data scale, the benefit comes from the consultation pattern itself (a structural shortcut), not from the stored content.

### Memory + depth are not additive

Adding external memory to an 8-layer baseline produces the same val loss as the 6-layer baseline without memory. The two mechanisms (more layers and memory consultation) provide the same kind of benefit — a secondary information pathway — and don't stack.

## Architecture

### SimpleMem (`src/simple_memory_transformer.py`)

A standard GPT-style causal transformer with one addition: a cross-attention layer inserted at the middle of the network. This layer attends to a bank of learned vectors (the "memory") using the hidden states as queries. The output is mixed into the residual stream through a learned gate. The memory vectors are initialized from the token embedding table for semantic diversity.

### Hierarchical Memory (`src/hierarchical_memory.py`)

Extends SimpleMem by moving the memory bank off-GPU. Memory vectors are stored as a numpy memmap file (disk-backed) with a FAISS index for fast approximate nearest-neighbor search. An LRU cache in CPU RAM keeps recently accessed vectors ready. At each forward pass, a retrieval query (mean-pooled hidden states) fetches the K most relevant vectors from the index, loads them to VRAM, and passes them through the same cross-attention mechanism as SimpleMem.

### Offload Transformer (`src/offload_transformer.py`)

A transformer whose forward and backward passes run entirely on GPU, but whose optimizer step offloads Adam states to CPU RAM. During the optimizer step, each block's momentum and variance tensors are loaded from CPU to VRAM, the Adam update runs on GPU, then the states are sent back to CPU. This keeps VRAM usage bounded by `model_weights + gradients + activations` without the 2x overhead of optimizer states.

Adaptive placement at initialization measures available VRAM and keeps as many blocks' optimizer states resident as possible. Gradient checkpointing trades recomputation for memory, allowing large batch sizes relative to model depth.

## Reproducing

### Setup

```bash
git clone <repo-url> && cd Nature
pip install -r requirements.txt
```

The training data is expected at `D:/DaBrainRecurrent/data/` — a collection of French text files (physics, history, cooking, conversations, ~12 MB total) with an 8K BPE tokenizer. Adjust `DataConfig.data_dir` and `DataConfig.tokenizer_path` in `src/config.py` to point to your own data and tokenizer.

### Baseline (6-layer transformer, ~5 min)

```bash
python experiments/run_baseline.py 42
```

### SimpleMem vs baseline (external memory, ~30 min)

```bash
python experiments/run_ablation_simple_mem.py
```

### Memory size sweep (~3 hours)

```bash
python experiments/run_simple_mem_sweep.py
```

### Hierarchical memory — Phase 1 validation (~1 hour)

```bash
python experiments/run_hierarchical_phase1.py
```

### Large model offloading — Phase 2 (~5 hours)

```bash
python experiments/run_phase2_final.py
```

Results are saved to `runs/<experiment_name>/log.json`. Summary CSV files are in `results/`.

## Limitations

- **Dataset size.** All experiments use 2.6M tokens of French text. This is sufficient to validate mechanisms but too small for large models to reach their potential. The memory consultation pattern and offloading mechanics are validated; their value at scale remains undemonstrated.

- **Optimizer throughput.** The block-by-block optimizer step (CPU<->GPU state transfers via PCIe) takes ~8 seconds for a 1B model, compared to ~500ms for the forward+backward pass. GPU utilization during the optimizer step is near zero. Industrial solutions (DeepSpeed ZeRO-Infinity, FSDP) solve this with fused kernels, NVMe offloading, and overlapped communication. This project uses naive PyTorch transfers.

- **Single-GPU only.** The offloading strategy is designed for one GPU. No distributed training was attempted.

- **French-only corpus.** Results may vary on other languages or domains, though the mechanisms are language-agnostic.

- **No comparison with industrial tools.** Libraries like DeepSpeed and HuggingFace Accelerate implement similar offloading strategies with far more optimization. This project demonstrates the principle from scratch, not a production-ready solution.

## Project structure

```
Nature/
  src/
    transformer.py              # Baseline GPT-style causal LM
    simple_memory_transformer.py # Transformer + learned memory cross-attention
    hierarchical_memory.py       # Memory with FAISS index on disk/RAM
    retrieval_memory.py          # FAISS retrieval-augmented transformer
    offload_transformer.py       # Layer offloading + CPU optimizer
    activation_field.py          # (Failed) bio-inspired activation field
    afm_transformer.py           # (Failed) activation field memory transformer
    config.py                    # Dataclasses for all configurations
    data.py                      # Data loading with quality filtering
    train.py                     # Training loop with validation
  experiments/                   # Runnable scripts for each experiment
  results/                       # Exported CSV summaries
  runs/                          # Training logs (JSON) and checkpoints
  journal.md                     # Full research log with hypotheses and results
  decision.md                    # Architectural decisions and their rationale
  echec.md                       # Honest post-mortem on the activation field failure
```

## Journal

The file `journal.md` contains the complete chronological log of the project: 10 experiments on the activation field and memory mechanisms, the pivot to hierarchical offloading, the optimizer optimization saga, and every dead end along the way. It's the unedited research narrative, preserved as documentation of the process.

## License

MIT
