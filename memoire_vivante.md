# Memoire vivante — etat de la branche

*Travail en cours. Le code est teste, les experiences ne sont pas encore concluantes.*

## L'idee, en trois etapes

1. **Un service ou la cave est indispensable.** Les experiences 1-10 testaient la memoire sur un corpus que le modele connaissait deja par coeur : rien a y chercher. Ici, la cave contient ce que les poids ne peuvent pas avoir : le debut du document en cours, sorti de la fenetre.
2. **Un foyer par token, causal, qui persiste.** Chaque token interroge la cave avec sa propre requete Q (celle de la self-attention, donc entrainee). La cave ne contient que les segments deja lus ; elle persiste de segment en segment et n'est videe qu'au changement de document. Corrige les deux defauts de `hierarchical_memory.py` : requete moyennee sur toute la sequence (fuite du futur) et `retrieval_proj` jamais entrainee (requete detachee).
3. **Le champ d'activation comme caviste.** Il ne pilote plus l'attention. Il observe ce que le modele a vraiment consulte, propage l'activation aux fiches voisines dans le temps, et choisit les fiches a monter en VRAM pour le pas suivant. Mesure : taux de succes contre une LRU de meme capacite. Il ne change pas le calcul (teste).

## Fichiers

| Fichier | Role |
|---|---|
| `src/knn_memory.py` | `KNNMemory` (la cave, K/V passes en RAM, anneau par flux), `Caviste`, `MemoryState` |
| `src/memorizing_transformer.py` | GPT causal dont une couche lit aussi la cave (porte apprise par tete) ; `use_memory=False` = controle a parametres identiques |
| `src/stream_data.py` | Lecture document par document ; categories de tokens `nouveau / proche / loin / hors_cave` |
| `src/synthetic_recall.py` | Tache "banquet avec allergie" : faits cle -> valeur, questions proches et lointaines |
| `src/stream_train.py` | Boucle d'entrainement en flux, evaluation par categorie |
| `experiments/run_synthetic_recall.py` | Tache synthetique, avec/sans cave, apparie par seed |
| `experiments/run_memoire_vivante.py` | Vrai corpus, avec/sans cave, apparie par seed, loss par categorie |
| `tests/test_memoire_vivante.py` | 8 tests : causalite, cave lue, reset par flux, caviste neutre, gradient vers Q |

## Etat

- Tests : 8/8 passent (`python tests/test_memoire_vivante.py`).
- Tache synthetique (CPU 2 coeurs, 4L/128d) : apres 2000 steps, le modele n'a pas encore appris a retrouver les faits, meme proches (precision ~3%, le hasard). Le mecanisme de recopie (induction) ne s'est pas encore forme. Un run plus long est en cours. Sans lui, la cave ne peut pas etre utilisee : la recherche n'est jamais recompensee au debut. C'est le meme probleme d'amorcage que le champ d'activation.
- Vrai corpus : pas encore lance (donnees sur D:). A faire sur la 3060 :

```bash
python tests/test_memoire_vivante.py
python experiments/run_synthetic_recall.py --steps 6000
python experiments/run_memoire_vivante.py
```

## Ce qui trancherait

- Synthetique : precision "rappel_loin" avec cave >> sans cave (~3%).
- Corpus : gain concentre sur les tokens `loin`, pas sur `nouveau`. Un gain uniforme serait un simple effet de capacite, comme SimpleMem.
