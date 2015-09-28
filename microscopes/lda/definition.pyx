# cython: embedsignature=True

cdef class model_definition:
    def __cinit__(self, n, v):
        self.n = n
        self.v = v
        self._thisptr.reset(new c_model_definition(n, v))