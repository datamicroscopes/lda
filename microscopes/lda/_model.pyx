# cython: embedsignature=True
from cython.operator cimport dereference as deref

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
        self._thisptr = c_initialize(defn._thisptr.get()[0], 0.2, 0.01, 0.5, _data, r._thisptr[0])
        print "Create state"


    def inference(self):
        deref(self._thisptr.get()).inference()

    def perplexity(self):
        print "perplexity3"
        # print deref(self._thisptr.get()).__class__
        if self._thisptr.get() == NULL:
            print "could not properly construct state"
            return 0

        self._thisptr.get().perplexity()
        return 10
        # return deref(self._thisptr.get()).perplexity()
        # return deref(self._thisptr.get()).perplexity()

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