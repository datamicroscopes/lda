# cython: embedsignature=True
from cython.operator cimport dereference as deref
from copy import deepcopy

cdef class state:
    """The underlying state of an HDP-LDA
    You should not explicitly construct a state object.
    Instead, use :func:`initialize`.
    Notes
    -----
    This class is not meant to be sub-classed.
    """
    def __cinit__(self, model_definition defn, vector[vector[size_t]] data, rng r, **kwargs):
        cdef vector[vector[size_t]] _data = deepcopy(data)
        self._thisptr = c_initialize(defn._thisptr.get()[0], .1, .5, .1, _data, r._thisptr[0])

    def perplexity(self):
        return self._thisptr.get().perplexity()

    def nentities(self):
        return self._thisptr.get().nentities()

    def ntopics(self):
        return self._thisptr.get().ntopics()

    def assignments(self):
        return self._thisptr.get()[0].assignments()

    def dish_assignments(self):
        return self._thisptr.get()[0].dish_assignments()

    def table_assignments(self):
        return self._thisptr.get()[0].table_assignments()

    def document_distribution(self):
        return self._thisptr.get()[0].document_distribution()

    def word_distribution(self, rng r):
        raise NotImplementedError()
        return self._thisptr.get()[0].word_distribution()

    def score_assignment(self):
        raise NotImplementedError()
        return self._thisptr.get()[0].score_assignment()

    def score_data(self, rng r):
        raise NotImplementedError()
        return self._thisptr.get()[0].score_data(r._thisptr[0])


def bind(state s, **kwargs):
    pass

def initialize(model_definition defn,
               vector[vector[size_t]] data,
               rng r,
               **kwargs):
    """Initialize state to a random, valid point in the state space
    Parameters
    ----------
    defn : model definition
    data : variadic dataview
    rng : random state
    """
    return state(defn=defn, data=data, r=r, **kwargs)