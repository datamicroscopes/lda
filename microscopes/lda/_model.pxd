from libcpp.vector cimport vector
from libc.stddef cimport size_t

from microscopes.common.variadic._dataview cimport (
    abstract_dataview,
)
from microscopes.common._rng cimport rng
from microscopes.lda._model_h cimport (
    state as c_state,
    document_model as c_document_model,
    table_model as c_table_model,
    initialize as c_initialize,
)
from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.lda.definition cimport (
    model_definition,
)

cdef class state:
    cdef shared_ptr[c_state] _thisptr

    # XXX: the type/structure information below is not technically
    # part of the model, and we should find a way to remove this
    # in the future
    cdef model_definition _defn


cdef class document_model:
    cdef shared_ptr[c_document_model] _thisptr


cdef class table_model:
    cdef shared_ptr[c_table_model] _thisptr
