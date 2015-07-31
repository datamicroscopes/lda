"""Test helpers specific to LDA

"""

import numpy as np
import itertools as it

from microscopes.common.testutil import permutation_iter


def toy_dataset(defn):
    """Generate a toy variadic dataset for HDP-LDA

    """
    avg_doc_len = 50
    lengths = 1 + np.random.poisson(lam=avg_doc_len, size=defn.n)
    def mkrow(nwords):
        return list(np.random.choice(range(defn.v), size=nwords))
    return [mkrow(length) for length in lengths]


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