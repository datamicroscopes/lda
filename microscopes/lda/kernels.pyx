# cython: embedsignature=True
import numpy as np

from scipy import stats
from scipy.special import digamma


def lda_crp_gibbs(state s, rng r):
    """Gibbs transition kernel for LDA state object. Modifies
    state object in place.

    Implementation of "Posterior sampling in the Chinese restaurant
        franchise" as described in Teh et al (2005).
    """
    c_lda_crp_gibbs(s._thisptr.get()[0], r._thisptr[0])

def sample_gamma(state s, rng r, float a, float b):
    # Sample Dirichlet process disperson parameter gamma according to
    # Gregor Heinrich's scheme seen here: http://bit.ly/1baZ3zf
    # Follow (Escobar+West95) with n = T
    c_sample_gamma(s._thisptr.get()[0], r._thisptr[0], a, b)

def sample_alpha(state s, rng r, float a, float b):
    # Sample Dirichlet process disperson parameter gamma according to
    # Gregor Heinrich's scheme seen here: http://bit.ly/1baZ3zf
    # Follow (Teh+06)
    c_sample_alpha(s._thisptr.get()[0], r._thisptr[0], a, b)

def sample_beta(state s, rng r, float a, float b, int num_iterations=1000):
    # Blindly follow Heinrich's methodology: http://www.arbylon.net/projects/IldaGibbs.java (line 679)

    i = 0
    alpha = s.beta
    alpha0 = 0
    prec = 1 ** -5
    for _ in range(num_iterations):
        summk = 0
        summ = 0
        for tid in s.active_topics():
            summ += digamma(s.n_k(tid))
            for word_id in range(s.nwords()):
                summk += digamma(s.n_kv(tid, word_id))
        summ -= s.ntopics() * digamma(s.nwords() * alpha)
        summk -= s.ntopics() * s.nwords() * digamma(alpha)
        assert not np.isnan(summk)
        alpha = (a - 1 + alpha * summk) / (b + s.ntopics() * summ)
        if abs(alpha - alpha0) < prec:
            break
        else:
            alpha0 = alpha

        if i == num_iterations - 1:
            raise Exception("sample_beta did not converge.")
    s.beta = alpha
