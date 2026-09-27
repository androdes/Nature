"""
Le banquet avec allergie : une tache ou la cave est indispensable.

Chaque document est une longue suite de "remplissage" (une chaine de
Markov simple, apprenable), dans laquelle on glisse des faits
    FAIT  cle  valeur
puis, pour chaque fait, deux questions
    QUESTION  cle  valeur
  - une question PROCHE, quelques tokens apres le fait : le fait est encore
    sur le comptoir (meme segment), l'attention locale suffit ;
  - une question LOINTAINE, au moins un segment plus tard : le fait est
    sorti de la fenetre, il faut le retrouver dans la cave.

Les associations cle -> valeur sont tirees au hasard A CHAQUE document :
aucun poids ne peut les apprendre. Sans cave, la precision sur les
questions lointaines reste au hasard (1/n_values).

Pourquoi des questions proches ? C'est le meme savoir-faire (retrouver la
cle, recopier ce qui la suivait) que le modele apprend d'abord sur le
comptoir. La cave est lue avec les memes requetes Q et les memes cles K :
le savoir-faire appris localement se transfere a la cave. Sans ce
tremplin, la recherche dans la cave n'est jamais recompensee au debut
(le bon fait n'est pas dans le top-K), et rien ne s'apprend : c'est
exactement le probleme d'amorcage du champ d'activation (echec.md).
"""

import numpy as np
import torch

CAT_NAMES = {0: 'remplissage', 1: 'rappel_proche', 2: 'rappel_loin'}


class RecallTask:

    def __init__(self, seq_len=64, n_segments=12, n_filler=48, n_keys=32,
                 n_values=32, n_facts=16, task_seed=1234):
        self.T = seq_len
        self.L = seq_len * n_segments + 1
        self.n_filler = n_filler
        self.FAIT = n_filler
        self.QUESTION = n_filler + 1
        self.key0 = n_filler + 2
        self.val0 = self.key0 + n_keys
        self.n_keys, self.n_values = n_keys, n_values
        self.vocab_size = self.val0 + n_values
        self.n_facts = n_facts

        # Chaine de Markov du remplissage : 4 successeurs par token, fixes
        rng = np.random.default_rng(task_seed)
        self.succ = np.stack([rng.choice(n_filler, 4, replace=False) for _ in range(n_filler)])
        self.succ_p = np.array([0.55, 0.25, 0.12, 0.08])

    def _filler(self, rng, n):
        out = np.empty(n, dtype=np.int64)
        out[0] = rng.integers(self.n_filler)
        choices = rng.choice(4, size=n, p=self.succ_p)
        for i in range(1, n):
            out[i] = self.succ[out[i - 1], choices[i]]
        return out

    def sample(self, rng):
        """
        Retourne (doc, cats) : cats[p] = 1 (rappel proche) ou 2 (rappel
        lointain) si doc[p] est une valeur a retrouver, 0 sinon.
        """
        T, L = self.T, self.L
        doc = self._filler(rng, L)
        cats = np.zeros(L, dtype=np.int64)
        busy = np.zeros(L, dtype=bool)

        def free(p):
            return p >= 1 and p + 3 <= L and not busy[max(p - 1, 0):p + 4].any()

        keys = rng.choice(self.n_keys, self.n_facts, replace=False)
        vals = rng.integers(self.n_values, size=self.n_facts)
        for k, v in zip(keys, vals):
            key, val = self.key0 + k, self.val0 + v
            # Le fait : dans la premiere moitie du document
            for _ in range(1000):
                p = int(rng.integers(1, L // 2))
                if free(p):
                    break
            else:
                continue
            doc[p:p + 3] = [self.FAIT, key, val]
            busy[p:p + 3] = True
            # Question proche : 4 a 28 tokens apres le fait
            # Question lointaine : au moins un segment + 16 tokens plus tard
            for lo, hi in ((p + 4, p + 28), (p + T + 16, L - 3)):
                for _ in range(1000):
                    q = int(rng.integers(lo, max(hi, lo + 1)))
                    if free(q):
                        doc[q:q + 3] = [self.QUESTION, key, val]
                        busy[q:q + 3] = True
                        seg_start = ((q + 2 - 1) // T) * T   # entrees du segment de la cible
                        cats[q + 2] = 1 if p >= seg_start else 2
                        break
        return doc, cats


class SyntheticStreams:
    """Meme interface que DocumentStreams, avec des documents toujours neufs."""

    def __init__(self, task: RecallTask, n_streams, seed=0):
        self.task = task
        self.B = n_streams
        self.T = task.T
        self.rng = np.random.default_rng(seed)
        self.docs = [None] * n_streams
        self.pos = [0] * n_streams

    def next_batch(self):
        T = self.T
        x = np.zeros((self.B, T), dtype=np.int64)
        y = np.zeros((self.B, T), dtype=np.int64)
        reset = np.zeros(self.B, dtype=bool)
        for b in range(self.B):
            if self.docs[b] is None or self.pos[b] + T + 1 > len(self.docs[b]):
                self.docs[b], _ = self.task.sample(self.rng)
                self.pos[b] = 0
                reset[b] = True
            p = self.pos[b]
            x[b], y[b] = self.docs[b][p:p + T], self.docs[b][p + 1:p + T + 1]
            self.pos[b] += T
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(reset)
