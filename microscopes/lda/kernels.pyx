# cython: embedsignature=True
import numpy as np

from scipy import stats


def lda_crp_gibbs(state s, rng r):
<<<<<<< HEAD
    c_lda_crp_gibbs(s._thisptr.get()[0], r._thisptr[0])

def lda_sample_dispersion(state s, rng r):
    # Sample Dirichlet process disperson parameters according to
    # Gregor Heinrich's scheme seen here: http://bit.ly/1baZ3zf
    T = sum([len(s.tables(eid)) - 1 for eid in range(s.nentities())])

    # Hyper-hyper params copied from Heinrich: http://bit.ly/1gyiIuE
    num_samples = 10  # R
    aalpha = 5
    balpha = 0.1
    abeta = 0.1
    bbeta = 0.1
    bgamma = 0.1  # ?
    agamma = 5  # ?

    for _ in range(num_samples):
        # gamma: root level (Escobar+West95) with n = T
        eta = stats.beta(s.gamma + 1, T).rvs()
        bloge = bgamma - np.log(eta)
        K = s.ntopics()
        pie = 1. / (1. + (T * bloge / (agamma + K - 1)))
        u = stats.bernoulli(pie).rvs()
        s.gamma = stats.gamma(agamma + K - 1 + u, 1. / bloge).rvs()

        # alpha: document level (Teh+06)
        qs = 0.
        qw = 0.
        for doc in s._data:
            doc_len = len(doc)
            qs += stats.bernoulli(doc_len * 1. / (doc_len + s.alpha)).rvs()
            qw += np.log(stats.beta(s.alpha + 1, doc_len).rvs())
        s.alpha = stats.gamma(aalpha + T - qs, 1. / (balpha - qw)).rvs()

    # state = update_beta(state, abeta, bbeta)
=======
    """Gibbs transition kernel for LDA state object. Modifies
    state object in place.

    Implementation of "Posterior sampling in the Chinese restaurant
        franchise" as described in Teh et al (2005).
    """
    c_lda_crp_gibbs(s._thisptr.get()[0], r._thisptr[0])
>>>>>>> master
