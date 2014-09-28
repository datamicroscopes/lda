from microscopes.lda._model_h cimport (
    table_model,
    document_model,
)
from microscopes.common._random_fwd_h cimport rng_t


cdef extern from "microscopes/lda/kernels.hpp":
    void assign \
        "microscopes::kernels::gibbsT::assign<microscopes::lda::table_model>" \
        (table_model &, rng_t &) except +

    void assign2 \
        "microscopes::kernels::gibbsT::assign2<microscopes::lda::document_model>" \
        (document_model &, rng_t &) except +
