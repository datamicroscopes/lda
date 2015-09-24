from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.stddef cimport size_t

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.common._random_fwd_h cimport rng_t


cdef extern from "microscopes/lda/model.hpp" namespace "microscopes::lda":
    cdef cppclass model_definition:
        model_definition(size_t, size_t) except +

    cdef cppclass state:
        double perplexity()
        size_t nentities()
        size_t ntopics()
        size_t nwords()
        size_t nterms(size_t) except +

        string serialize() except +
        vector[vector[size_t]] assignments()
        vector[vector[size_t]] dish_assignments()
        vector[vector[size_t]] table_assignments()
        vector[vector[float]] document_distribution()
        vector[map[size_t, float]] word_distribution()

        float score_assignment()
        float score_data(rng_t &)


cdef extern from "microscopes/lda/model.hpp" namespace "microscopes::lda::state":
    shared_ptr[state] \
    initialize(const model_definition &defn,
        float alpha, float beta, float gamma,
        size_t initial_dishes,
        vector[vector[size_t]] &docs,
        rng_t & rng) except +

    shared_ptr[state] \
    initialize_explicit "microscopes::lda::state::initialize" (
        const model_definition &defn,
        float alpha, float beta, float gamma,
        const vector[vector[size_t]] &dish_assignments,
        const vector[vector[size_t]] &table_assignments,
        vector[vector[size_t]] &docs,
        rng_t &rng) except +