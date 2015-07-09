from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.stddef cimport size_t

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.common._random_fwd_h cimport rng_t
from microscopes.common.variadic._dataview_h cimport dataview


cdef extern from "microscopes/lda/model.hpp" namespace "microscopes::lda":
    cdef cppclass model_definition:
        model_definition(size_t, size_t) except +

    cdef cppclass state:
        double perplexity()
        size_t nentities()
        size_t ntopics()
        size_t nwords()
        size_t nterms(size_t) except +

        vector[map[size_t, size_t]] dish_assignments()
        vector[vector[size_t]] table_assignments()

        float score_assignment()
        float score_data(rng_t &)





cdef extern from "microscopes/lda/model.hpp" namespace "microscopes::lda::state":
    shared_ptr[state] initialize(const model_definition &,
                                 float,
                                 float,
                                 float,
                                 vector[vector[size_t]] &,
                                 rng_t &) except +
