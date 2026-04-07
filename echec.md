# Echec — Rapport de resultat negatif

## These testee
Un mecanisme de memoire a champ d'activation distribue avec foyer mobile peut compenser l'echelle : un transformer de N params avec ce mecanisme bat un transformer de N params sans, d'au moins 5%.

## Verdict
**These non validee.** 10 experiences, 0 gain >= 5%. Le maximum observe est ~1.3%, equivalent a ajouter 2 couches transformer, et les effets ne sont pas additifs.

## Les 10 experiences

| # | Modele | Params | Val Loss | vs 6L |
|---|--------|--------|----------|-------|
| 1 | Baseline 6L | 6.85M | 4.711 +/- 0.052 | ref |
| 1 | Baseline 8L | 8.43M | 4.648 +/- 0.010 | -1.3% |
| 3 | AFMT v1 (full attn) | 8.24M | 4.696 +/- 0.038 | -0.3% |
| 4 | AFMT v2 (focus K=64) | 8.43M | 4.763 +/- 0.056 | +1.1% |
| 5 | AFMT tiny (2L+mem) | 2.03M | 4.999 +/- 0.057 | N/A |
| 6 | SimpleMem (6L+512) | ~7.2M | 4.646 +/- 0.057 | -1.4% |
| 7 | Sweep N/layers | ~7.2M | 4.670 +/- 0.053 | -0.9% |
| 8 | Retrieval FAISS | ~7.2M | 4.651 +/- 0.103 | -1.3% |
| 9 | Corpus memory (77K chunks) | ~7.2M | 4.648 +/- 0.097 | -1.3% |
| 10 | 8L + SimpleMem | ~8.8M | 4.710 +/- 0.075 | -0.0% |

## Ce qui a echoue et pourquoi

### Le champ d'activation (exps 3-5)
Le mecanisme bio-inspire — spreading activation, decay, foyer mobile — est reste **inerte** pendant tout l'entrainement. L'activation std n'a jamais depasse 0.03. Causes :
- **Bootstrap circulaire** : les activations ont besoin d'attention focalisee pour se differencier, mais l'attention a besoin d'activations differenciees pour se focaliser
- **Softmax sur N elements** : dilue l'attention a 1/N par element, signal trop faible pour piloter les dynamiques d'activation
- **Le top-K selection** coupe les gradients pour 97% de la memoire

### Le scaling de la memoire (exps 7-9)
N=128 ≈ N=512 ≈ N=8000 ≈ N=50000. Le contenu de la memoire ne compte pas — qu'il soit appris (SimpleMem), des embeddings bruts (retrieval), ou des representations reelles du corpus (corpus memory). Le gain vient du pattern structurel de consultation, pas de l'information stockee.

### L'additivite (exp 10)
8L + memoire = 6L. La memoire et les couches supplementaires sont **substitutives**, pas additives. Elles offrent le meme benefice : un chemin supplementaire pour l'information. Empiler les deux ne sert a rien.

## Ce qui a fonctionne (malgre l'echec de la these)

### Le pattern de consultation
Un transformer peut apprendre a consulter une memoire externe via cross-attention. Ce comportement emerge naturellement (gate apprise a ~0.5) et fournit un benefice reel de ~1.3%. C'est equivalent en performance a 2 couches transformer supplementaires, mais avec 5x moins de parametres supplementaires (~330K vs 1.6M).

### Configuration optimale trouvee
- 1 seule cross-attention, placee au milieu du transformer
- 512 elements de memoire suffisent (128 presque autant)
- Plus de layers de consultation NUIT (L1 > L2 > L3)
- Le transformer prefere une intervention minimale

## Pourquoi le seuil de 5% n'est pas atteint

L'hypothese la plus probable : **le dataset est trop petit** (2.6M tokens) pour que la memoire externe apporte du contenu que le transformer ne peut pas deja encoder dans ses poids. Avec 6.8M params pour 2.6M tokens (2.6 params/token), le baseline peut quasi-memoriser le corpus. La memoire externe n'a rien de substantiel a ajouter.

Pour tester cette hypothese, il faudrait : un corpus 10-100x plus grand (26-260M tokens), un modele de meme taille (6.8M params), et refaire les experiences. Le ratio passerait a 0.026-0.26 params/token, regime ou le baseline ne peut plus tout memoriser.

## Ce que je ferais avec plus de ressources

1. **Plus de donnees** : Wikipedia FR complet (~500M tokens). Ratio 0.014 params/token. Dans ce regime, la memoire externe contient de l'information que le transformer ne peut pas encoder dans ses poids. Le retrieval de chunks de corpus devrait enfin montrer un gain proportionnel a N.

2. **Memoire ecrivable** : le SimpleMem actuel est en lecture seule pendant le forward pass. Un mecanisme d'ecriture (comme le DNC) permettrait au transformer de deposer de l'information dans la memoire et de la relire plus tard. Cela teste la persistance inter-sequence, absente des experiences actuelles.

3. **Le champ d'activation, autrement** : l'echec du champ d'activation vient du bootstrap circulaire, pas du concept. Une approche alternative : ne pas utiliser l'activation pour PILOTER l'attention, mais pour STRUCTURER la memoire a posteriori. Le champ d'activation serait alors un mecanisme d'organisation (clustering dynamique des elements memoire), pas un mecanisme de selection.

4. **Evaluation qualitative** : la perplexite ne capture peut-etre pas ce que la memoire apporte. Une evaluation sur des taches de factual recall (QA), de coherence long-range, ou de generalisation a de nouveaux domaines pourrait reveler des benefices invisibles dans la loss globale.

## Conclusion
Le champ d'activation bio-inspire ne fonctionne pas dans sa forme actuelle. Mais le transformer a quelque chose d'inattendu : il sait developper un comportement de consultation d'une memoire externe, et ce comportement a une valeur structurelle reelle (~1.3%). Ce n'est pas l'architecture visee au depart, mais c'est un resultat exploitable — et peut-etre un socle pour construire, un jour, quelque chose qui ressemble davantage a la pensee qui circule.
