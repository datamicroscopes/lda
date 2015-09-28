# cython: embedsignature=True

def lda_crp_gibbs(state s, rng r):
    """Gibbs transition kernel for LDA state object. Modifies
    state object in place.

    Implementation of "Posterior sampling in the Chinese restaurant
        franchise" as described in Teh et al (2005).
    """
    c_lda_crp_gibbs(s._thisptr.get()[0], r._thisptr[0])