# cython: embedsignature=True


# python imports
from microscopes.common import validator


cdef class model_definition:
    def __cinit__(self, n, v):
        validator.validate_positive(n, "n")
        validator.validate_positive(v, "v")

        self._n = n
        self._v = v
        self._thisptr.reset(new c_model_definition(n, v))
