from microscopes.lda._kernels_h cimport lda_crp_gibbs as c_lda_crp_gibbs
from microscopes.lda._kernels_h cimport sample_gamma as c_sample_gamma
from microscopes.lda._kernels_h cimport sample_alpha as c_sample_alpha
from microscopes.common._rng cimport rng
from microscopes.lda._model cimport state