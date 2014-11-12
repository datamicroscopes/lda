from libcpp.vector cimport vector
from libc.stddef cimport size_t

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.common._random_fwd_h cimport rng_t
from microscopes.common.variadic._dataview_h cimport dataview
from microscopes.common._typedefs_h cimport hyperparam_bag_t


cdef extern from "microscopes/lda/model.hpp" namespace "microscopes::lda":
    cdef cppclass model_definition:
        model_definition(size_t, size_t) except +

    cdef cppclass state:
        size_t nentities()
        size_t ntopics()
        size_t nwords()
        size_t nterms(size_t) except +
        float score_assignment()
        float score_data(rng_t &)

    cdef cppclass document_model:
        document_model(const shared_ptr[state] &,
                       const shared_ptr[dataview] &)

    cdef cppclass table_model:
        table_model(const shared_ptr[state] &, size_t)


cdef extern from "microscopes/lda/model.hpp" namespace "microscopes::lda::state":
    shared_ptr[state] \
    initialize(const model_definition &,
               const hyperparam_bag_t &,
               const hyperparam_bag_t &,
               const dataview &,
               size_t,
               const vector[vector[size_t]] &,
               rng_t &) except +

    shared_ptr[state] \
    initialize_explicit "microscopes::lda::state::initialize" (
               const model_definition &,
               const hyperparam_bag_t &,
               const hyperparam_bag_t &,
               const dataview &,
               const vector[vector[size_t]] &,
               const vector[vector[size_t]] &,
               rng_t &) except +
