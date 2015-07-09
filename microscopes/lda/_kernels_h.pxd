from _model_h cimport state
from microscopes.common._random_fwd_h cimport rng_t

cdef extern from "microscopes/lda/kernels.hpp":
    void lda_crp_gibbs  "microscopes::kernels::lda_crp_gibbs" (state &, rng_t &)