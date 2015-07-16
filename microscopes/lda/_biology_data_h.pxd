from _model_h cimport state
from libcpp.vector cimport vector
from microscopes.common._random_fwd_h cimport rng_t

cdef extern from "microscopes/lda/biology_data.hpp":
    vector[vector[size_t]] docs "data::docs"