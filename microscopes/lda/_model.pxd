from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libc.stddef cimport size_t

from microscopes.common.variadic._dataview cimport (
    abstract_dataview,
)
from microscopes.common._rng cimport rng
from microscopes.lda._model_h cimport (
    state as c_state,
    initialize as c_initialize,
)
from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.lda.definition cimport model_definition


cdef class state:
    """The underlying state of a Hierarchial Dirichlet Process LDA

    You should not explicitly construct a state object.
    Instead, use `initialize`.
    """
    cdef shared_ptr[c_state] _thisptr
    # cdef model_definition _defn
    # cdef abstract_dataview _data