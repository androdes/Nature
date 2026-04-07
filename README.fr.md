# Nature — Transformer a memoire hierarchique

Entrainer un transformer d'un milliard de parametres sur un GPU de 12 Go en repartissant les poids entre VRAM, RAM et disque. Une demonstration que la VRAM est une contrainte de vitesse, pas une contrainte de capacite.

## L'idee

La communaute ML traite la VRAM comme une limite absolue sur la taille des modeles. Si un modele ne tient pas en VRAM, il faut un plus gros GPU ou du calcul distribue. Mais la VRAM n'est que le niveau le plus rapide d'une hierarchie memoire. La RAM est 10x plus grande et 10x plus lente. Le disque est 100x plus grand et 100x plus lent. Un systeme qui utilise les trois niveaux peut entrainer des modeles bien au-dela de ce que la VRAM seule permet — ca prend juste plus de temps.

Ce projet demontre ce compromis sur du materiel grand public. Un transformer de 1 milliard de parametres s'entraine sur une RTX 3060 (12 Go de VRAM) : ses poids vivent en VRAM, les etats de l'optimiseur transitent par la RAM CPU, et le gradient checkpointing limite la memoire des activations. Le modele converge correctement. C'est lent (8 secondes par step au lieu de 0.5), mais ca marche.

La tortue bat le lievre — pas en vitesse, mais en distance parcourue.

## Ce qui a ete demontre

### 1. La consultation d'une memoire externe fonctionne

Ajouter une couche de cross-attention entre un transformer et une petite memoire externe (512 vecteurs appris) ameliore la loss de validation de ~1.3% par rapport a un transformer pur avec le meme backbone. C'est equivalent en qualite a ajouter deux couches transformer supplementaires, mais pour 5x moins de parametres additionnels (~330K au lieu de 1.6M).

La decouverte surprenante : le *contenu* de la memoire n'a pas d'importance. Des parametres appris, des embeddings de tokens, des chunks de corpus recuperes par FAISS, ou des vecteurs aleatoires produisent tous le meme gain de ~1.3%. Ce qui aide, c'est le pattern structurel de consultation — une deuxieme voie d'information a cote de la self-attention — pas l'information qui y est stockee.

| Modele | Parametres | Val Loss | vs Baseline |
|--------|-----------|----------|-------------|
| Baseline 6L/256d | 6.85M | 4.711 +/- 0.052 | reference |
| Baseline 8L/256d | 8.43M | 4.648 +/- 0.010 | -1.3% |
| SimpleMem (6L + 512 memoire) | ~7.2M | 4.646 +/- 0.057 | -1.4% |
| Retrieval FAISS (K=64) | ~7.2M | 4.651 +/- 0.103 | -1.3% |

Tous les resultats sont moyennes sur 3 seeds aleatoires. Donnees d'entrainement : 2.6M tokens de texte francais diversifie.

### 2. La hierarchie memoire preserve la qualite

Quand la memoire externe vit partiellement en RAM CPU au lieu d'etre entierement en VRAM, la qualite d'apprentissage est indistinguable de la version tout-VRAM. Ecart mesure : 0.1% (dans le bruit). L'index FAISS gere le retrieval depuis la RAM de maniere transparente, et un cache LRU garde les elements frequemment accedes proches du GPU.

### 3. Un modele de 1 milliard de parametres s'entraine sur 12 Go de VRAM

En combinant gradient checkpointing et un optimiseur bloc par bloc qui garde les etats Adam en RAM CPU, un transformer 36 couches / 1536 dimensions (1.03 milliard de parametres) s'entraine sur une RTX 3060. Pic d'utilisation VRAM : 8.3 Go pour un modele dont les poids seuls totalisent 3.9 Go (le reste : activations et gradients). Les etats de l'optimiseur (7.8 Go pour le momentum et la variance d'Adam) vivent entierement en RAM CPU et sont charges en VRAM un bloc a la fois pendant le step d'optimisation.

Le modele converge — la loss de validation descend regulierement de 6.3 a 5.1 en 2000 steps. Il n'atteint pas la qualite du baseline de 6.8M parametres (4.71) parce que les donnees d'entrainement sont trop petites (2.6M tokens pour 1 milliard de parametres), pas parce que le mecanisme echoue.

| Modele | Parametres | Pic VRAM | ms/step | Val Loss |
|--------|-----------|----------|---------|----------|
| Baseline 6L | 6.8M | ~2 Go | ~200 | 4.711 |
| 350M tout-VRAM | 353M | 6.9 Go | ~500 | 4.875 |
| 1B offloade | 1 030M | 8.3 Go | ~8 000 | 5.116 |

### 4. Le goulot d'etranglement, ce sont les donnees, pas l'architecture

Avec 2.6M tokens d'entrainement, meme un modele de 6.8M parametres overfit (train loss 2.0 vs val loss 4.7). Les modeles plus gros overfittent plus vite. La memoire externe ne peut pas aider parce qu'il n'y a rien de nouveau a recuperer — le transformer memorise deja l'integralite du corpus dans ses poids. Pour que le scaling de la memoire devienne rentable, il faudrait 100M+ tokens.

## Ce qui n'a pas marche

### L'Activation Field Memory Transformer

L'architecture d'origine etait bio-inspiree : un "champ" de valeurs d'activation reparti sur les elements memoire, avec un foyer mobile qui se deplace a travers le champ en propageant l'activation aux voisins semantiques. L'idee etait de simuler la circulation d'une pensee — quand on pense "chat", "souris" et "jardin" s'allument par association.

Dix experiences ont teste des variantes de ce mecanisme :
- Attention sur tous les elements memoire (ecart-type de l'activation : 0.0007 — essentiellement zero)
- Selection top-K du foyer (ecart-type : 0.03, mais statique pendant tout l'entrainement)
- Routage du foyer par contenu
- Differentes strategies d'initialisation et hyperparametres

Aucune n'a produit de differentiation significative dans le champ d'activation. Le mecanisme etait inerte. La cause profonde : un probleme d'amorcage circulaire. Les activations ont besoin d'attention focalisee pour se differencier, mais l'attention a besoin d'activations differenciees pour se focaliser. Les deux commencent uniformes et restent uniformes.

L'ablation qui a revele cela : retirer entierement le champ d'activation et ne garder que la cross-attention vers la memoire (SimpleMem) a produit de *meilleurs* resultats que le mecanisme complet. Les dynamiques d'activation n'etaient pas seulement inutiles — elles interferaient avec l'apprentissage.

### Le scaling de la memoire ne paye pas

Augmenter le nombre d'elements memoire de 128 a 50 000 ne produit aucune amelioration. A cette echelle de donnees, le gain provient du pattern de consultation lui-meme (un raccourci structurel), pas du contenu stocke.

### Memoire et profondeur ne s'additionnent pas

Ajouter une memoire externe a un baseline de 8 couches produit la meme val loss qu'un baseline de 6 couches sans memoire. Les deux mecanismes (plus de couches et consultation memoire) apportent le meme type de benefice — une voie d'information secondaire — et ne se cumulent pas.

## Architecture

### SimpleMem (`src/simple_memory_transformer.py`)

Un transformer causal standard de type GPT avec un ajout : une couche de cross-attention inseree au milieu du reseau. Cette couche fait de l'attention sur une banque de vecteurs appris (la "memoire") en utilisant les hidden states comme requetes. La sortie est injectee dans le flux residuel a travers une gate apprise. Les vecteurs memoire sont initialises depuis la table d'embeddings des tokens pour garantir une diversite semantique initiale.

### Memoire hierarchique (`src/hierarchical_memory.py`)

Etend SimpleMem en deplacant la banque memoire hors du GPU. Les vecteurs sont stockes dans un fichier numpy memmap (sur disque) avec un index FAISS pour la recherche approximative de plus proches voisins. Un cache LRU en RAM CPU garde les vecteurs recemment accedes prets a l'emploi. A chaque forward pass, une requete de retrieval (mean pooling des hidden states) recupere les K vecteurs les plus pertinents depuis l'index, les charge en VRAM, et les passe dans le meme mecanisme de cross-attention que SimpleMem.

### Offload Transformer (`src/offload_transformer.py`)

Un transformer dont les passes forward et backward tournent entierement sur GPU, mais dont le step d'optimisation decharge les etats Adam en RAM CPU. Pendant le step, les tenseurs de momentum et de variance de chaque bloc sont charges depuis le CPU vers la VRAM, la mise a jour Adam tourne sur GPU, puis les etats repartent vers le CPU. Cela maintient l'utilisation VRAM a `poids + gradients + activations` sans le surcout 2x des etats de l'optimiseur.

Le placement adaptatif a l'initialisation mesure la VRAM disponible et garde autant d'etats d'optimiseur residants que possible. Le gradient checkpointing echange du recalcul contre de la memoire, permettant des tailles de batch plus grandes relativement a la profondeur du modele.

## Reproduire les resultats

### Installation

```bash
git clone <repo-url> && cd Nature
pip install -r requirements.txt
```

Les donnees d'entrainement sont attendues dans `D:/DaBrainRecurrent/data/` — une collection de fichiers texte francais (physique, histoire, cuisine, conversations, ~12 Mo) avec un tokenizer BPE 8K. Ajuster `DataConfig.data_dir` et `DataConfig.tokenizer_path` dans `src/config.py` pour pointer vers vos propres donnees et tokenizer.

### Baseline (transformer 6 couches, ~5 min)

```bash
python experiments/run_baseline.py 42
```

### SimpleMem vs baseline (memoire externe, ~30 min)

```bash
python experiments/run_ablation_simple_mem.py
```

### Sweep taille memoire (~3 heures)

```bash
python experiments/run_simple_mem_sweep.py
```

### Memoire hierarchique — validation Phase 1 (~1 heure)

```bash
python experiments/run_hierarchical_phase1.py
```

### Offloading gros modeles — Phase 2 (~5 heures)

```bash
python experiments/run_phase2_final.py
```

Les resultats sont sauvegardes dans `runs/<nom_experience>/log.json`. Les fichiers CSV de synthese sont dans `results/`.

## Limitations connues

- **Taille du dataset.** Toutes les experiences utilisent 2.6M tokens de texte francais. C'est suffisant pour valider les mecanismes mais trop petit pour que les gros modeles atteignent leur potentiel. Le pattern de consultation memoire et la mecanique d'offloading sont valides ; leur valeur a grande echelle reste a demontrer.

- **Debit de l'optimiseur.** Le step bloc par bloc (transferts d'etats CPU<->GPU via PCIe) prend ~8 secondes pour un modele de 1 milliard de parametres, contre ~500ms pour le forward+backward. L'utilisation GPU pendant le step est proche de zero. Des solutions industrielles (DeepSpeed ZeRO-Infinity, FSDP) resolvent cela avec des kernels fusionnes, de l'offloading NVMe, et de la communication chevauchee. Ce projet utilise des transferts PyTorch naifs.

- **Mono-GPU uniquement.** La strategie d'offloading est concue pour un seul GPU. Aucun entrainement distribue n'a ete tente.

- **Corpus francais uniquement.** Les resultats peuvent varier sur d'autres langues ou domaines, bien que les mecanismes soient agnostiques a la langue.

- **Pas de comparaison avec les outils industriels.** Des bibliotheques comme DeepSpeed et HuggingFace Accelerate implementent des strategies d'offloading similaires avec bien plus d'optimisation. Ce projet demontre le principe en partant de zero, pas une solution prete pour la production.

## Structure du projet

```
Nature/
  src/
    transformer.py              # Baseline GPT causal
    simple_memory_transformer.py # Transformer + cross-attention vers memoire apprise
    hierarchical_memory.py       # Memoire avec index FAISS sur disque/RAM
    retrieval_memory.py          # Transformer augmente par retrieval FAISS
    offload_transformer.py       # Offloading de couches + optimiseur CPU
    activation_field.py          # (Echec) champ d'activation bio-inspire
    afm_transformer.py           # (Echec) transformer a champ d'activation
    config.py                    # Dataclasses de configuration
    data.py                      # Chargement de donnees avec filtrage qualite
    train.py                     # Boucle d'entrainement avec validation
  experiments/                   # Scripts reproductibles pour chaque experience
  results/                       # Syntheses CSV exportees
  runs/                          # Logs d'entrainement (JSON) et checkpoints
  journal.md                     # Journal de recherche complet
  decision.md                    # Decisions architecturales et leurs raisons
  echec.md                       # Post-mortem honnete sur le champ d'activation
```

## Journal de recherche

Le fichier `journal.md` contient le log chronologique complet du projet : 10 experiences sur le champ d'activation et les mecanismes de memoire, le pivot vers l'offloading hierarchique, la saga d'optimisation de l'optimiseur, et chaque impasse rencontree en chemin. C'est le recit de recherche non edite, conserve comme documentation du processus.

## Licence

MIT
