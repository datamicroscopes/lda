# cython imports
from libcpp.vector cimport vector
from libc.stddef cimport size_t

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes._models cimport _base
from microscopes.lda._model_h cimport (
    model_definition as c_model_definition,
)


cdef class model_definition:
    cdef shared_ptr[c_model_definition] _thisptr
    cdef readonly int _n
    cdef readonly int _v
