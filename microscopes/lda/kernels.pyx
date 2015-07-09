# cython: embedsignature=True

def lda_crp_gibbs(state s):
    c_lda_crp_gibbs(s._thisptr.get()[0])