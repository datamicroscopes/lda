"""Test helpers specific to LDA

"""

import numpy as np


def toy_dataset(defn):
    lengths = 1 + np.random.poisson(lam=1.0, size=defn.n())
    def mkrow(nwords):
        return np.random.choice(range(defn.v()), size=nwords)
    return map(mkrow, lengths)
