# Journal de recherche - Nature

## 2026-04-06 - Session 1 : Demarrage

### Etat du repo
Repo vide. Projet DaBrainRecurrent existant avec donnees et tokenizer reutilisables.

### Donnees disponibles
- Source : D:/DaBrainRecurrent/data/
- ~12MB, ~108K lignes
- Contenu : texte francais diversifie (physique, cuisine, histoire, conversations, knowledge, tasks)
- corpus.txt est en anglais (Gutenberg) - a filtrer ou inclure selon resultats
- Tokenizer BPE 8K existant (dabrain_bpe_8k.json)

### Decisions de session
1. Reutiliser le tokenizer BPE 8K existant plutot que d'en entrainer un nouveau
2. Filtrer les donnees : supprimer lignes avec trop de caracteres speciaux, HTML, signes parasites
3. Commencer par un baseline transformer minimal pour avoir une reference solide
4. Architecture baseline : 6 layers, 256 dim, 4 heads, FFN 1024, weight tying -> ~6.8M params

### Experience 0 : Baseline convergence check (500 steps)
- **Hypothese** : le baseline transformer converge sur les donnees francaises en 500 steps
- **Setup** : 6L/256d/4H, 6.8M params, batch=64, seq=256, lr=3e-4 cosine, seed=42
- **Donnees** : 37 fichiers, 8.7M chars, 2.6M tokens, 9709 train seqs, 511 val seqs
- **Attendu** : loss qui descend de ~9 (log(8000)) a ~4-5 en 500 steps
- **Obtenu** : train 8.79->4.02, val 7.71->5.11. Debit ~88K tok/s. Val loss encore en descente lente.
- **Interpretation** : Convergence confirmee. Gap train/val (~1 point) = overfitting attendu sur petit dataset. Le modele genere du francais semi-coherent ("La chaleur transforme le liquide en gaz"). Le cosine schedule atteint le LR min trop tot pour 500 steps. Pour le full run : 5000 steps, warmup 200.
- **Run** : D:/Nature/runs/baseline_s42_500steps/

### Experience 1 : Baseline complet (5000 steps, 3 seeds)
- **Hypothese** : avec plus de steps, val loss descend sous 5.0
- **Setup** : 6L/256d/4H, 6.8M params, batch=64, 5000 steps, seeds 42/137/256
- **Attendu** : val loss ~4.5-5.0
- **Obtenu** : s42=4.760 (step 1500), s137=4.638 (step 1750), s256=4.734 (step 1750). **Moyenne: 4.711 +/- 0.052**.
- **Interpretation** : overfitting severe apres step ~1500. Best val ~4.71. Le modele memorise par brute force dans les poids (train loss ~2.0 vs val ~4.7). C'est notre reference a battre.
- **Runs** : D:/Nature/runs/baseline_full_s42/, baseline_full_s137/, baseline_full_s256/

### Observation methodologique importante
L'AFMT a 8.8M params vs 6.8M pour le baseline. Pour une comparaison equitable, il faut aussi un baseline a ~8.8M params (8 layers au lieu de 6). Sinon on pourrait attribuer un gain au mecanisme alors qu'il vient simplement des params supplementaires.

### Experience 2 : AFMT Phase 2 flat - quick check (20 steps)
- **Hypothese** : le forward pass et les gradients fonctionnent
- **Setup** : 6L/256d + 4096 elements memoire, 8.8M params, batch=32, 20 steps, lr=1e-3
- **Obtenu** : loss 9.04->7.00 en 20 steps. Gradients OK sur memory, resting_logits, neighbor_weights. Activation std passe de 0.000002 a 0.000385 (differentiation lente mais reelle). Gates a ~0.12.
- **Interpretation** : le mecanisme fonctionne. Les gates commencent petit (sigmoid(-2)=0.12) ce qui permet au transformer de converger d'abord. Les activations se differencient progressivement. Pret pour le full training.

### Experience 3 : AFMT v1 full attention (3 seeds, 5000 steps)
- **Hypothese** : la memoire avec attention sur tous les N elements ameliore la loss
- **Setup** : 6L + 2048 elements, attention sur tout N, gate=-2, batch=32, 5000 steps
- **Obtenu** : val_loss 4.696 +/- 0.038. Activations std=0.0007, PLAT. Aucune differentiation.
- **Interpretation** : ECHEC. Softmax sur 2048 elements = attention trop diluee. Le champ d'activation ne se differencie jamais. Le mecanisme est inerte.

### Diagnostic : probleme de bootstrap
Le champ d'activation a un probleme circulaire :
- Les activations ont besoin d'attention focalisee pour se differencier
- L'attention a besoin d'activations differenciees pour se focaliser
- Les deux commencent uniformes et restent uniformes

Corrections tentees : gate 0.5 (au lieu de 0.12), zero-centering du biais, init memoire depuis token embeddings, gamma 5.0.

### Experience 4 : AFMT v2 FocusCrossAttention K=64 (3 seeds, 5000 steps)
- **Hypothese** : top-K selection (K=64) + selection content-based brise le bootstrap
- **Setup** : 6L + 2048 elements, focus_k=64, selection par pertinence + activation, gate=0.5, batch=32
- **Obtenu** : val_loss **4.763 +/- 0.056**. Pire que baseline 6L (4.711) et 8L (4.648).
- **Activations** : std=0.030 (mieux, mais PLAT tout le long). Le foyer selectionne des elements differents selon l'input, mais le champ ne s'adapte pas au fil de l'entrainement.
- **Interpretation** : ECHEC. Le mecanisme nuit. Possible cause : avec 2.6M tokens et 6.8M params (2.6 params/token), le baseline memorise deja tout. La memoire externe n'a rien de plus a apporter. Les params supplementaires du mecanisme causent de l'overfitting sans benefice.

### Tableau recapitulatif

| Modele | Params | Val Loss | Delta vs 6L |
|--------|--------|----------|-------------|
| Baseline 6L | 6.85M | 4.711 +/- 0.052 | reference |
| Baseline 8L | 8.43M | 4.648 +/- 0.010 | -1.3% |
| AFMT v1 | 8.24M | 4.696 +/- 0.038 | -0.3% |
| AFMT v2 K=64 | 8.43M | 4.763 +/- 0.056 | +1.1% |

### Consultation persona : sceptique methodologique
**Question** : le test est-il valide?
**Reponse** : NON. Avec 2.6 params/token, on teste la memoire sur un regime ou le baseline peut deja tout memoriser. Le test n'est pas equitable. Pour tester si "la memoire compense l'echelle", il faut un regime ou le modele est SOUS-DIMENSIONNE par rapport aux donnees.

### Experience 5 : Regime sous-dimensionne (2L/128d)
- **Hypothese** : avec 0.56 params/token, le modele ne peut pas memoriser. La memoire aide.
- **Setup** : 2L/128d (1.45M), AFMT 2L+mem (2.03M), baseline 4L (1.85M), batch=128
- **Obtenu** : 2L=5.017, AFMT=4.999 (-0.4%), 4L=4.853 (-3.3%)
- **Interpretation** : meme en regime sous-dim, l'AFMT n'aide presque pas. Ajouter des couches est 8x plus efficace.

### Experience 6 : Ablation - memoire simple sans activation field
- **Hypothese** : si on enleve le champ d'activation et qu'on garde juste la cross-attention a la memoire, est-ce que ca aide?
- **Setup** : 6L/256d + 512 elements memoire, 1 seule cross-attention, gate=0.5, init depuis token embeddings
- **Obtenu** :
  - SimpleMem tiny (2L): 4.997 vs baseline 2L 5.017 (-0.4%)
  - **SimpleMem large (6L): 4.646 vs baseline 6L 4.711 (-1.4%)** ← PREMIER SIGNAL POSITIF
  - Compare a baseline 8L (4.648) : performance equivalente avec 1.2M params en moins
- **Interpretation** : **DECOUVERTE CLE**. La memoire externe simple AIDE. Le champ d'activation etait le probleme, pas la memoire. 512 elements + 1 cross-attn = aussi efficace que +2 couches transformer, mais avec 5x moins de params supplementaires. L'activation field ajoutait de la complexite qui interferait avec l'apprentissage.

### Tableau recapitulatif mis a jour

| # | Modele | Params | Val Loss | Delta vs 6L |
|---|--------|--------|----------|-------------|
| 1 | Baseline 6L | 6.85M | 4.711 +/- 0.052 | reference |
| 1 | Baseline 8L | 8.43M | 4.648 +/- 0.010 | -1.3% |
| 3 | AFMT v1 full attn | 8.24M | 4.696 +/- 0.038 | -0.3% |
| 4 | AFMT v2 focus K=64 | 8.43M | 4.763 +/- 0.056 | +1.1% |
| 5 | AFMT tiny | 2.03M | 4.999 +/- 0.057 | N/A |
| 6 | **SimpleMem large** | **~7.2M** | **4.646 +/- 0.057** | **-1.4%** |

### Experience 7 : Sweep memoire simple (N, n_layers)
- **Hypothese** : plus de memoire ou plus de cross-attn layers aide
- **Setup** : 6L/256d + SimpleMem, configs: N={128,256,512,1024} x L={1,2,3}
- **Obtenu** :

| Config | Val Loss | Delta vs 6L |
|--------|----------|-------------|
| n128_L1 | 4.682 +/- 0.019 | -0.6% |
| n256_L1 | 4.723 +/- 0.040 | +0.3% |
| **n512_L1** | **4.670 +/- 0.053** | **-0.9%** |
| n1024_L1 | 4.702 +/- 0.033 | -0.2% |
| n512_L2 | 4.733 +/- 0.005 | +0.5% |
| n512_L3 | 4.775 +/- 0.013 | +1.4% |

- **Interpretation** :
  1. Le gain plafonne a N=512. N=1024 est pire (overfitting des params memoire)
  2. Plus de cross-attn layers NUIT systematiquement (L1 < L2 < L3)
  3. Le transformer prefere une intervention minimale : 1 consultation, pas 3
  4. 128 elements suffisent presque (-0.6%) — le gain vient du pattern de consultation, pas de la quantite stockee

### Relecture (suite a observation de l'humain)
Le SimpleMem n'est pas une version degradee de l'AFMT. C'est un transformer qui a developpe un COMPORTEMENT DE CONSULTATION. La memoire externe est un canal d'information distinct des self-attention. Le gain ne vient pas de la taille de la memoire mais de l'existence de ce canal.

Cela reouvre la hierarchie disque/RAM/VRAM — non pas pour le champ d'activation, mais pour permettre a ce canal de consulter un corpus plus grand. Un retrieval FAISS permet N arbitraire sans impact VRAM.

### Experience 8 : Retrieval FAISS - scaling N
- **Hypothese** : le gain scale avec N si le contenu compte
- **Setup** : 6L/256d + retrieval cross-attn (1 layer, K=64 recuperes), N={512, 8000, 50000}
- **Obtenu** :
  - N=512:   4.651 +/- 0.103 (-1.3%)
  - N=8000:  4.663 +/- 0.097 (-1.0%)
  - N=50000: 4.652 +/- 0.091 (-1.3%)
- **Interpretation** :
  1. Le retrieval BAT la SimpleMem parametrique (4.651 vs 4.670). La selection sharp (K=64 via FAISS) aide plus que le contenu appris.
  2. **N ne scale PAS** : 512 = 8K = 50K. Le contenu ne compte pas.
  3. Retrieval ≈ baseline 8L (4.651 vs 4.648), mais avec 200K params supplementaires au lieu de 1.6M.
  4. Variance haute (~0.10) vs baseline 8L (0.01) : le retrieval est moins stable.

**Note** : biais methodologique — l'index FAISS est construit a l'init et jamais mis a jour. Les embeddings evoluent pendant le training mais l'index reste fige. Malgre ca, le retrieval fonctionne.

### Experience 9 : Memoire de corpus (77K chunks reels)
- **Hypothese** : du vrai contenu semantique (hidden states du corpus) depasse le plafond de 1.3%
- **Setup** : baseline 6L pre-entraine encode le corpus -> 77K chunks de 32 tokens -> FAISS -> retrieval K=64
- **Obtenu** : 4.648 +/- 0.097 (-1.3%)
- **Interpretation** : meme plafond. Le contenu reel ne fait pas mieux que des embeddings aleatoires. Confirme : le contenu ne compte pas, le pattern oui.

### Experience 10 : 8L + SimpleMem (test d'additivite)
- **Hypothese** : memoire + couches supplementaires sont additifs
- **Setup** : 8L/256d + SimpleMem 512 elements, 1 cross-attn
- **Obtenu** : 4.710 +/- 0.075 (-0.0% vs 6L, PIRE que 8L seul)
- **Interpretation** : ECHEC. Les effets sont **substitutifs**. Memoire et couches font la meme chose : un chemin supplementaire pour l'information. Empiler les deux ne sert a rien.

---

## Synthese finale — 10 experiences

| # | Modele | Val Loss | vs 6L |
|---|--------|----------|-------|
| 1 | Baseline 6L | 4.711 +/- 0.052 | ref |
| 1 | Baseline 8L | 4.648 +/- 0.010 | -1.3% |
| 3 | AFMT v1 full attn | 4.696 +/- 0.038 | -0.3% |
| 4 | AFMT v2 focus K=64 | 4.763 +/- 0.056 | +1.1% |
| 5 | AFMT tiny 2L+mem | 4.999 +/- 0.057 | N/A |
| 6 | SimpleMem 6L+512 | 4.646 +/- 0.057 | -1.4% |
| 7 | Sweep best (n512_L1) | 4.670 +/- 0.053 | -0.9% |
| 8 | Retrieval FAISS | 4.651 +/- 0.103 | -1.3% |
| 9 | Corpus memory 77K | 4.648 +/- 0.097 | -1.3% |
| 10 | 8L + SimpleMem | 4.710 +/- 0.075 | -0.0% |

**Seuil de 5% (4.475) jamais atteint. echec.md redige.**

Voir echec.md pour l'analyse complete, les causes d'echec, et les pistes avec plus de ressources.

---

## Pivot : La tortue et le lievre — hierarchie VRAM/RAM/disque

### Phase 1 : la mecanique tient
- SimpleMem VRAM (A) : 4.670 +/- 0.053
- Hierarchique N=512 (B) : 4.663 +/- 0.096
- Ecart : 0.1% → **PHASE 1 VALIDEE**

### Phase 2 : rotation des couches + optimizer offload

#### Autocritique : sous-utilisation GPU (2026-04-07)

L'humain a observe le GPU a 11% d'utilisation moyenne pendant l'entrainement. Deux questions posees. Reponses honnetes :

**Question 1 : est-ce que je savais que le GPU etait sous-utilise ?**

Oui et non. J'avais les donnees. Le benchmark du 1B montre :
- compute : 36-50ms (forward+backward)
- optimizer : 3277-6374ms
- ratio GPU utile : ~1%

J'ai imprime ces chiffres. J'ai meme ecrit "GPU util ~1%". Mais je n'ai pas reagi proportionnellement. J'ai traite ca comme un detail a optimiser plus tard, alors que c'est le probleme principal. J'ai vu le chiffre sans en tirer la consequence : un systeme a 1% d'efficacite GPU n'est pas "lent mais acceptable", c'est un systeme ou 99% du temps est gaspille.

La raison : j'etais en mode "faire marcher la mecanique" et pas en mode "faire marcher efficacement". Une fois le forward+backward+optimizer fonctionnel sans crash, j'ai considere le probleme resolu et je suis passe a la convergence. C'est une erreur de priorisation. L'efficacite aurait du etre un pre-requis, pas une optimisation ulterieure.

**Question 2 : est-ce que j'avais prevu l'inefficacite ?**

Non. Quand j'ai mis prefetch=False "pour valider la mecanique de base", je n'avais pas chiffre l'impact. J'ai raisonne qualitativement ("ca sera plus lent, on optimisera apres") sans estimer de combien. Si j'avais fait le calcul :

- 36 blocs * 48MB/bloc * 2 (aller-retour) = 3.5 GB de transferts par step
- PCIe 3.0 ~10 GB/s theorique, ~5 GB/s effectif = 700ms juste pour les transferts
- Plus l'overhead Python, la serialisation, les syncs = probablement 2-3s
- Contre 50ms de compute = ratio ~2-5% d'utilisation GPU

Ce calcul aurait du me faire reagir AVANT le premier run. Je ne l'ai pas fait. J'ai prefere "essayer et voir" plutot que "estimer et planifier".

**Ce que ca revele sur mon processus :**
1. Je regarde les metriques quand elles apparaissent mais je ne les ANTICIPE pas
2. Je traite les goulots d'etranglement comme des problemes sequentiels (d'abord la correctness, puis la performance) alors qu'un goulot a 99% aurait du bloquer le pipeline
3. Quand l'humain m'a dit "tortue avec turbo" et "le GPU attend", j'ai corrige en reactif au lieu d'avoir identifie le probleme proactivement

**Pour la suite :** avant de lancer un run, estimer le temps par step et l'utilisation GPU attendue. Si < 50%, optimiser AVANT de lancer.

#### Optimisation de l'optimizer offload

| Version | Total/step | Optimizer | GPU util | Methode |
|---------|-----------|-----------|----------|---------|
| V1 model.to(cpu) | 39s | 39s | ~0% | round-trip complet GPU<->CPU |
| V2 shadow CPU | 6.4s | 6.0s | ~1% | shadow params CPU, single step |
| V3 block GPU | 3.3s | 2.9s | ~0% | step GPU par bloc (etait SGD fallback!) |
| V4 36 opts CPU | 12.7s | 12.2s | ~4% | 36 optimizers separees |
| V5 single opt CPU | 12.7s | 12.2s | ~4% | 1 optimizer, tous les shadow |
| V3 restored | 6.6-8.3s | 6-7.7s | ~7-8% | block GPU avec vrai Adam |

Le CPU Adam sur 1B params est irremediablement lent (~6-12s). Le GPU Adam par bloc est limite par les transfers PCIe (36 * 96MB * 2 = 6.9 GB). Pas de solution miracle sans changer la bande passante materielle.

### Phase 2 : resultats de convergence

#### 350M tout-VRAM (12L/1536d, batch=8, 5000 steps)
- best_val: 4.875 (plateau atteint)
- peak_vram: 6.9 GB
- GPU: 100%, ~500ms/step
- Train loss 3.5 vs val 4.9 = overfitting severe (2.6M tokens insuffisants)

#### 1B offload (36L/1536d, batch=2, 2000 steps)
- best_val: 5.116 (encore en descente)
- peak_vram: 8.3 GB
- GPU: ~8%, ~8s/step (optimizer-bound)
- Pas encore converge (batch=2 trop petit, pas assez de steps)

#### Bilan phase 2
- **La mecanique fonctionne** : un 1B tourne sur RTX 3060, converge sans crash
- **Le goulot est le dataset** (2.6M tokens), pas l'architecture
- **Le goulot de performance est l'optimizer** (PCIe transfers pour les Adam states)
- Le 350M tout-VRAM est l'optimum pratique sur cette carte : assez gros pour demonstrer le scaling, assez petit pour 100% GPU util
