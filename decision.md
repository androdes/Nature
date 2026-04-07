# Decisions architecturales - Nature

## 2026-04-06 : Synthese litterature et choix fondamentaux

### Litterature inspectee

| Source | Ce qu'on prend | Ce qu'on ecarte | Raison |
|--------|---------------|-----------------|--------|
| Memory Networks (Weston 2014) | Memoire externe differentiable, lectures douces | Attention sur TOUTE la memoire a chaque pas | Viole la hierarchie - pas de foyer |
| DNC (Graves 2016) | Ecriture/lecture structuree, liens temporels | Complexite du mecanisme complet, acces global | Trop lourd, pas de hierarchie |
| RAG/REALM/RETRO | Retrieval depuis corpus externe pendant inference | Retrieval one-shot, non-differentiable | Pas de propagation, pas de persistance d'activation |
| FAISS/HNSW | Index vectoriel pour recherche approximative rapide | - | C'est notre couche disque |
| Spreading Activation (Collins & Loftus 1975) | **COEUR** : activation qui se propage le long d'aretes ponderees | - | C'est exactement l'intuition du foyer mobile |
| Hopfield Networks modernes (Ramsauer 2020) | Fondation theorique pour memoire associative continue | Calcul global (energy sur tout le reseau) | Trop couteux, mais la theorie valide notre approche |
| Mixture of Experts | Activation sparse, routing | Experts statiques, pas charges dynamiquement | Inspire le routing mais pas la memoire |
| Product Key Memory (Lample 2019) | Memoire massive avec lookup rapide par produit de cles | - | Inspire notre mecanisme RAM avec indexation rapide |

### Architecture choisie : Activation Field Memory Transformer (AFMT)

**Principe** : Un transformer standard augmente d'un champ d'activation distribue avec foyer mobile et hierarchie memoire disque/RAM/VRAM.

**Composants** :

1. **Champ d'activation** : N vecteurs de dimension d, chacun avec un scalaire d'activation a_i in (epsilon, 1]. Jamais zero, jamais binaire.

2. **Foyer** : A chaque pas, le hidden state du transformer produit une requete de focus. Attention douce sur les K2 elements en VRAM. Le foyer = les elements les plus actives apres cette attention.

3. **Propagation** : Graphe sparse appris G. Apres le focus, les elements focalises propagent l'activation a leurs voisins : a_j += alpha * a_i * w_ij. Decay global : a_i *= (1 - delta). Pas de mort (clamp a epsilon).

4. **Hierarchie** :
   - Disque : tous les N elements (FAISS index + vecteurs + activations)
   - RAM : top-K1 par activation (~10K elements), tenseurs CPU
   - VRAM : top-K2 par activation (~512 elements), tenseurs GPU
   - Promotions/demotions par seuils adaptatifs sur les activations

5. **Injection dans le transformer** : Cross-attention entre hidden states et elements VRAM focalises. Un layer supplementaire dans chaque bloc transformer (ou tous les k blocs).

### Comptage de parametres

Decision : compter TOUS les parametres (y compris memoire sur disque) comme parametres totaux. Mais la metrique cle est la **performance par parametres actifs en VRAM**. Le baseline utilise tous ses params en VRAM. L'AFMT utilise moins de params en VRAM mais a acces a plus de connaissances.

Comparaison equitable :
- Baseline : transformer de X params, tous en VRAM
- AFMT : meme transformer de X params + mecanisme memoire (surcout VRAM petit) + memoire externe (sur disque)
- Montrer que l'AFMT bat le baseline malgre le meme backbone transformer

### Tokenizer

Decision : utiliser le tokenizer de CamemBERT (sentencepiece, 32K vocab, pre-entraine sur du francais). Pas besoin de re-entrainer un tokenizer, c'est du temps perdu.

### Corpus

Decision revisee : donnees francaises diversifiees de D:/DaBrainRecurrent/data/ (~12MB, 2.6M tokens). Plus petit que Wikipedia FR mais deja disponible et propre.

### Dimensions reelles (post-implementation)

- Baseline 6L : 6 layers, 256d, 4 heads, FFN 1024, weight tying -> 6,852,608 params
- Baseline 8L : 8 layers, meme config -> 8,432,128 params (controle equitable)
- AFMT : 6L transformer + 2048 elements memoire (256d), 32 voisins, 3 cross-attn layers -> 8,235,526 params
  - Reduit de N=4096 a N=2048 : avec batch=64 et N=4096, la cross-attention (B,H,T,N) saturait la VRAM a 12GB.
  - N=2048 batch=32 : peak 4.2GB, confortable sur RTX 3060
- Context length : 256 tokens
- Batch size : 32 (uniforme pour toutes les experiences, evite saturation VRAM)

### Comparaison equitable des parametres

La comparaison a trois niveaux :
1. **Baseline 6L vs AFMT** : le mecanisme memoire aide-t-il, meme avec plus de params?
2. **Baseline 8L vs AFMT** : a params comparables (~8.5M vs ~8.8M), la memoire bat-elle plus de couches?
3. **Params actifs en VRAM** : si l'AFMT bat le baseline 8L, c'est que la structure compte plus que la quantite.

Le baseline 8L est le controle critique. Sans lui, on ne peut pas distinguer "le mecanisme aide" de "plus de params aide".

### Differentiabilite

- Operations VRAM (attention, propagation intra-VRAM) : differentiables
- Transferts RAM<->VRAM : gating sigmoide doux sur (activation - seuil) -> differentiable
- Transferts Disque<->RAM : non-differentiable (prefetch heuristique) -> OK, pas dans le forward pass critique
- Lookup FAISS : non-differentiable, utilise uniquement pour l'acces disque

### Apprentissage du graphe de propagation

Decision de depart : cosine similarity dans l'espace d'embedding pour l'initialisation, puis edge weights ajustes par gradient pendant l'entrainement. Top-K_neighbors voisins par element (K_neighbors=32).

Alternative consideree : co-occurrence statistique. Ecartee car non-differentiable et necessite un pre-entrainement separe.

### Gestion de la latence disque

Decision : prefetch asynchrone. Un thread separe charge les elements predits depuis le disque pendant que le GPU calcule. Le cache RAM est dimensionne pour absorber ~100ms de latence I/O. En pratique sur SSD, le prefetch devrait etre quasi-instantane.

---

## 2026-04-06 : Relecture des resultats — changement de cadre

### Ce que les donnees disent

Le SimpleMem (6L + 512 elements, 1 cross-attn) obtient 4.646 — egal au baseline 8L (4.648) avec 1.2M params en moins. L'AFMT avec champ d'activation obtient 4.696-4.763, systematiquement pire.

Ma premiere lecture etait : "la memoire simple marche, le champ d'activation echoue". L'implication etait que le champ d'activation est un mecanisme casse qu'il faut reparer.

### Deuxieme lecture

Le SimpleMem n'est pas une version degradee de l'AFMT. C'est un objet different. Le transformer a appris a CONSULTER une memoire externe — a developper un pattern de lecture qui n'existe pas dans un transformer pur. C'est un mode de fonctionnement qualitativement distinct :

- Le transformer pur encode tout dans ses poids. L'attention porte sur les tokens du contexte.
- Le SimpleMem a deux sources d'information : les tokens du contexte (self-attention) et un reservoir externe (cross-attention). Il a appris quand consulter l'une ou l'autre.

Ce n'est pas "un transformer avec de la memoire en plus". C'est un transformer qui a developpe un comportement de consultation. La gate a 0.5 au depart, et le modele a CHOISI de l'utiliser plutot que de la fermer.

### Ce que cette lecture ouvre

Si le transformer sait consulter une memoire de 512 elements, rien ne l'oblige a rester a 512. La memoire, etant separee du traitement (pas dans les poids du transformer, pas dans la VRAM pendant le forward pass des self-attention), peut vivre ailleurs.

La hierarchie disque/RAM/VRAM redevient pertinente — mais pour une raison differente de celle du depart :

- **Raison initiale** : le champ d'activation bio-inspire NECESSITE une hierarchie pour simuler la pensee qui circule. La hierarchie sert le mecanisme cognitif.
- **Raison emergente** : un transformer qui sait consulter une memoire externe peut consulter une memoire PLUS GRANDE sans que le cout VRAM augmente. La hierarchie sert l'echelle.

Le champ d'activation reste une piste pour plus tard (donner une structure a la consultation), mais il n'est plus un prerequis. La base qui marche — le pattern de consultation — peut grandir seule.

### Questions experimentales que cette relecture ouvre

1. **Scaling de la memoire** : si 512 elements donnent -1.4%, que donnent 5K? 50K? 500K? A quel point le gain sature-t-il? Avec FAISS pour l'indexation, on peut monter a des millions d'elements sans que la VRAM augmente — seuls les K elements recuperes passent par la cross-attention.

2. **Memoire non-parametrique** : les 512 elements actuels sont des parametres appris (comme des embeddings). Et si la memoire contenait des DONNEES reelles — des chunks du corpus d'entrainement? Le transformer apprendrait alors a consulter un corpus, pas des abstractions. C'est le pont vers RAG, mais integre dans l'entrainement.

3. **Separation apprentissage/stockage** : dans le SimpleMem, la memoire est apprise conjointement avec le transformer. Mais on pourrait geler la memoire (embeddings pre-calcules du corpus) et n'apprendre que le mecanisme de consultation (les projections Q/K/V de la cross-attention). Cela teste : le transformer peut-il apprendre a lire une memoire qu'il n'a pas construite?

4. **Le champ d'activation comme structure de consultation** : si on a une memoire de 500K elements, on ne peut pas faire softmax sur 500K a chaque token. Il faut un mecanisme de selection. Le retrieval par FAISS est une option. Le champ d'activation en est une autre — mais cette fois, il ne pilote pas la memoire, il pilote la SELECTION. C'est un role plus modeste et potentiellement plus viable.

### Decision pour la suite

Je choisis la question 1 : **scaling de la memoire**. C'est le test le plus direct de l'hypothese emergente. Si le gain sature a 512, la consultation est un artefact. Si le gain continue avec N croissant, on tient quelque chose.

Implementation : remplacer la memoire parametrique (nn.Parameter) par un retrieval depuis un index FAISS. A chaque forward pass, le transformer produit une requete, FAISS retourne les K elements les plus proches, et la cross-attention opere sur ces K elements. La memoire peut alors etre arbitrairement grande sans impact sur la VRAM (seuls K elements sont charges).

C'est la hierarchie disque/RAM/VRAM du plan initial, mais arrivee par un chemin different. Pas parce qu'un champ d'activation l'exige, mais parce qu'un transformer qui consulte a besoin d'avoir quelque chose de grand a consulter.

Alternatives considerees :
- Question 2 (memoire non-parametrique) : interessante mais couple deux variables (taille + contenu). Mieux de varier la taille d'abord.
- Question 3 (memoire gelee) : teste une propriete secondaire. Plus tard.
- Question 4 (activation field comme selecteur) : premature tant qu'on n'a pas montre que le scaling aide.

---

## 2026-04-06 : Pivot — La tortue et le lievre

### Contexte
10 experiences terminees. Le champ d'activation echoue. La memoire simple donne ~1.3% mais plafonne. Le contenu ne compte pas — le gain vient du pattern de consultation. echec.md redige pour la these initiale.

Reorientation du projet. La vitesse n'est plus un critere. Ce qui compte : un transformer qui consulte une memoire physiquement repartie au-dela de la VRAM.

### Nouvelle these
A VRAM egale, un transformer avec une memoire etendue sur RAM + disque peut atteindre une qualite que la VRAM seule ne permet pas. La tortue (lente, mais avec acces a plus) bat le lievre (rapide, mais limite par la VRAM).

### Pourquoi c'est different des experiences precedentes
Les exps 7-9 ont montre que scaling N ne change rien (512 = 50K). Mais ces tests avaient un vice : le contenu etait redondant (embeddings repetes + bruit, ou chunks d'un corpus deja vu). La memoire "grande" ne contenait rien de plus que la memoire "petite".

Pour que l'extension ait un interet, il faut que les elements sur disque/RAM contiennent de l'information que ceux en VRAM ne contiennent PAS. Deux pistes :
1. **Memoire parametrique apprise** mais trop grande pour la VRAM : les elements sur disque sont des parametres du modele, mis a jour par gradient, mais jamais tous en VRAM en meme temps. Chaque batch ne charge que les elements pertinents.
2. **Memoire non-parametrique de corpus** : les elements sont des representations fixes du corpus. Le volume justifie l'extension — 500K chunks de 32 tokens = 16M tokens de contexte retrievable.

Je choisis la piste 1 pour la phase 1 (mecaniquement plus proche de SimpleMem) et je reserve la piste 2 pour la phase 2 si la mecanique tient.

### Architecture : HierarchicalMemoryTransformer

Base : SimpleMem (6L/256d, 1 cross-attn au milieu, gate apprise). Seule difference : la memoire est trop grande pour la VRAM.

**Trois niveaux :**
- **VRAM** : K elements (ex: 512). Ce sont les elements charges pour la cross-attention du batch courant. Selectionnes par FAISS retrieval sur la requete du batch.
- **RAM** : M elements (ex: 10K). Cache CPU en tenseurs PyTorch. Contient les elements recemment utilises + ceux precharges par anticipation.
- **Disque** : N elements total (ex: 50K-500K). Stockes comme numpy memmap + index FAISS. Acces en ~1ms sur SSD.

**Flux par batch :**
1. Le transformer traite les tokens jusqu'a la couche de cross-attention
2. Une requete de retrieval est produite (mean pool des hidden states)
3. FAISS cherche les K plus proches dans l'index (couvre tout N)
4. Les K elements sont charges en VRAM (depuis RAM si present, sinon depuis disque)
5. Cross-attention sur ces K elements
6. Le transformer continue
7. Apres le backward, les gradients des K elements charges sont accumules
8. Periodiquement, les gradients accumules sont appliques a tous les elements (sur CPU/disque)

**Gestion de la promotion/demotion :**
- Pas de promotion/demotion explicite. Le FAISS retrieval EST le mecanisme de selection.
- Le cache RAM garde les elements les plus recemment accedes (LRU).
- Le prefetch charge les voisins FAISS du batch courant en RAM avant le prochain batch.

**Entrainabilite :**
- Les elements memoire sont des parametres (requires_grad=True)
- Seuls les K elements charges par batch recoivent des gradients
- On accumule les gradients et on applique le step optimizer sur tous les elements periodiquement (tous les N_accum batches)
- Le FAISS index est reconstruit periodiquement (tous les R steps) car les elements evoluent

**Instrumentation :**
- Compteur d'acces par element (combien de fois chaque element est selectionne)
- Distribution des acces par niveau (VRAM/RAM/disque)
- Latence par step (temps de retrieval, temps de cross-attention, temps total)
- Couverture : quel % des elements totaux est accede au moins une fois par epoch

### Critere de succes phase 1
Version B (hierarchique) converge avec val_loss dans +/-5% de version A (tout en VRAM). Le temps par step peut etre 10x plus long.

### Critere de succes phase 2 (corrige)

L'ancien critere ("battre la VRAM en val loss") etait mal cadre. Le vrai argument : la VRAM a un plafond physique dur. Le disque non. L'architecture qui peut continuer a grandir quand la VRAM ne le peut plus est celle qui gagne a long terme. C'est le lievre et la tortue.

~~L'ancien setup (scaling memoire externe 10x-1000x) est abandonne. On sait deja que le contenu de la memoire externe plafonne — pousser sa taille verifie la mecanique, pas la these.~~

### Phase 2 reformulee : rotation des couches transformer VRAM/RAM

La phase 1 prouve que des parametres memoire peuvent vivre hors VRAM et etre utilises correctement. La phase 2 etend ce principe aux poids du transformer lui-meme — la chose la plus importante.

**Principe** : a tout instant, seules 1-2 couches transformer sont en VRAM. Les autres vivent en RAM. On charge les poids depuis la RAM avant le forward d'une couche, on les redescend apres. Pendant le backward, on remonte dans l'ordre inverse.

**Ce que ca change** : la VRAM cesse d'etre un plafond pour la taille du modele. Un transformer de 50M ou 100M params peut tourner sur une 3060 12GB en gardant seulement 1-2 couches actives a la fois. Plus lent, mais possible.

**Optimisations prevues** :
1. Prefetch asynchrone : pendant que la couche N calcule, precharger N+1 en parallele
2. Gradient checkpointing : sauvegarder quelques activations, recalculer les autres au backward
3. Granularite : couche par couche ou par groupes, a determiner empiriquement

**Ordre d'implementation** :
1. D'abord la rotation des blocs seule (sans memoire externe) — verifier que ca converge
2. Puis combiner rotation + memoire externe hierarchique — le systeme complet

Je commence par la rotation seule. C'est plus risque que la memoire externe (les poids du transformer sont critiques, pas optionnels) et doit etre valide en isolation avant d'ajouter la memoire.

**Critere de succes** :
- Entrainement converge avec rotation (val loss stable)
- VRAM des poids << total des poids (ex: 1 couche en VRAM pour un modele 12 couches)
- Val loss comparable a un baseline tout-VRAM de meme taille
- On entraine un modele PLUS GROS que ce que la VRAM permettrait sans rotation

**Critere d'echec** :
- Divergence
- Degradation massive (>20%) vs baseline tout-VRAM
- Temps par step impraticable meme pour un projet qui accepte la lenteur

**Note** : la technique existe (DeepSpeed, Accelerate sous "CPU offloading"). L'angle ici est different : pas un compromis desagreable mais une these architecturale — la VRAM comme fenetre glissante sur un modele plus grand.
