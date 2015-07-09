# cython: embedsignature=True

def lda_crp_gibbs(state s, rng r):
    c_lda_crp_gibbs(s._thisptr.get()[0], r._thisptr[0])