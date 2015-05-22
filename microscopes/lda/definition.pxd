from libcpp.vector cimport vector

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes._models cimport _base
from microscopes._models_h cimport model as c_component_model
from microscopes.lda._model_h cimport model_definition as c_model_definition
from microscopes.lda._model_h cimport state as c_state

cdef class model_definition:
    cdef shared_ptr[c_model_definition] _thisptr
    cdef readonly int _n
    cdef readonly list _models



