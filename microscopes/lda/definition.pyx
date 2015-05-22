# cython: embedsignature=True


from microscopes.models import model_descriptor
from microscopes.common import validator
import operator as op
import copy

cdef class model_definition:
    def __cinit__(self,
                  int n,
                  models):
        self._n = n
        self._models = []
        for model in models:
            self._models.append(model)

