# cython: embedsignature=True
import numpy as np

from scipy import stats


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
