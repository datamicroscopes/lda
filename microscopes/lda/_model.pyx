# cython: embedsignature=True


cdef class state:
    """The underlying state of an HDP-LDA
    You should not explicitly construct a state object.
    Instead, use :func:`initialize`.
    Notes
    -----
    This class is not meant to be sub-classed.
    """
    def __cinit__(self, model_definition defn, data, rng r, **kwargs):
        cdef vector[vector[size_t]] _data = data
        self._thisptr = c_initialize(defn._thisptr.get()[0], 0.1, 0.001, 0.1, _data, r._thisptr[0])



def bind(state s, **kwargs):
    pass

def initialize(model_definition defn,
               data,
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