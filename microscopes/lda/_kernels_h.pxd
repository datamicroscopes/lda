from _model_h cimport state
from microscopes.common._random_fwd_h cimport rng_t

cdef extern from "microscopes/lda/kernels.hpp":
    void lda_crp_gibbs  "microscopes::kernels::lda_crp_gibbs" (state &, rng_t &)
    void sample_gamma  "microscopes::kernels::lda_hyperparameters::sample_gamma" (state &, rng_t &, float, float)
    void sample_alpha  "microscopes::kernels::lda_hyperparameters::sample_alpha" (state &, rng_t &, float, float)