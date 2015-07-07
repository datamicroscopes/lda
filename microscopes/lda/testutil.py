"""Test helpers specific to LDA

"""

import numpy as np
import itertools as it

from microscopes.common.testutil import permutation_iter


def toy_dataset(defn):
    """Generate a toy variadic dataset for HDP-LDA

    """

    lengths = 1 + np.random.poisson(lam=1.0, size=defn.n())
    def mkrow(nwords):
        return np.random.choice(range(defn.v()), size=nwords)
    return map(mkrow, lengths)


def permutations(doclengths):
    """Generate a permutation of XXX

    WARNING: very quickly becomes intractable
    """

    perms = [permutation_iter(length) for length in doclengths]
    for prod in it.product(*perms):
        dishes = sum([max(x) + 1 for x in prod])
        for p in permutation_iter(dishes):
            idx = 0
            ret = []
            for d in prod:
                ntables = max(d) + 1
                ret.append(tuple(p[idx:idx+ntables]))
                idx += ntables
            yield prod, tuple(ret)