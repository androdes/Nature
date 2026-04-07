# Plan de recherche - Nature

## These
Un mecanisme de memoire a champ d'activation distribue avec foyer mobile et hierarchie disque/RAM/VRAM peut compenser l'echelle : un transformer de N params avec ce mecanisme bat un transformer de N params sans, a cout VRAM comparable.

## Phase 0 : Infrastructure (session 1) - FAIT
- [x] Synthese litterature -> decision.md
- [x] Baseline transformer (GPT causal, 6L/256d/4H, ~6.8M params, weight tying)
- [x] Data loader avec filtrage qualite sur corpus DaBrainRecurrent
- [x] Training loop avec validation, seeds, logging
- [x] Verification convergence sur 500 steps -> OK, loss 8.79->5.11

## Phase 1 : Baseline complet (session 1) - FAIT
- [x] Entrainement complet baseline 6L, 3 seeds x 5000 steps
- [x] Reference etablie : **val_loss = 4.711 +/- 0.052** (best step ~1500-1750)
- [x] Overfitting severe apres step 1500

## Phase 2 : Activation Field Memory v1 - flat (session 1) - EN COURS
Hypothese : le champ d'activation avec foyer mobile et propagation ameliore la loss a params comparables.

- [x] Implementer ActivationField + MemoryCrossAttention + AFMTransformer
- [x] Fix: softmax au lieu de sigmoid pour la propagation (evite saturation)
- [x] Verification: forward pass OK, gradients OK, activations se differencient
- [x] Preparer baseline 8L (8.4M params) comme controle equitable vs AFMT (8.8M)
- [ ] **EN COURS**: Entrainement baseline 8L (3 seeds) + AFMT (3 seeds)
- [ ] Comparer les 3 conditions: baseline 6L vs baseline 8L vs AFMT
- [ ] Documenter resultats dans journal.md

### Architectures en competition
| Modele | Params | Backbone | Extra |
|--------|--------|----------|-------|
| Baseline 6L | 6.85M | 6L/256d/4H | - |
| Baseline 8L | 8.43M | 8L/256d/4H | - |
| AFMT | 8.83M | 6L/256d/4H | 4096 mem elements + 3 cross-attn |

### Scenarios de resultats
- AFMT < 6L < 8L : victoire claire (memoire > couches, meme la 6L est battue)
- AFMT < 8L < 6L : victoire (memoire > couches, mais 8L overfit plus)
- 6L < AFMT < 8L : mitige (memoire aide vs couches, mais moins que le backbone nu)
- 6L < 8L < AFMT : echec phase 2 (memoire fait pire, pivot necessaire)

## Phase 3 : Hierarchie disque/RAM/VRAM (si Phase 2 positive)
Hypothese : la hierarchie permet d'augmenter massivement N sans augmenter le cout VRAM.

- [ ] Implementer MemoryHierarchy : FAISS sur disque, cache RAM, fenetre VRAM
- [ ] Implementer prefetch predictif base sur la propagation
- [ ] Gating differentiable pour les transferts RAM<->VRAM
- [ ] Entrainement avec N=100K elements, comparaison Phase 2 et baseline
- [ ] 3 seeds, documenter

## Phase 4 : Raffinement et ablations (si Phase 3 positive)
- [ ] Ablation : propagation vs pas de propagation
- [ ] Ablation : hierarchie vs flat
- [ ] Ablation : foyer mobile vs attention globale sur memoire
- [ ] Comparaison finale, 3 seeds

## Criteres
- Succes : >= 5% reduction de loss val ou loss equivalente avec 30% moins de params actifs
- Echec apres 10 experiences : rediger echec.md
- Un gain de 2% avec std 3% n'est PAS un gain
- Reference baseline 6L : **4.711 +/- 0.052**
- Seuil 5% : val_loss <= **4.475**
