# cython: embedsignature=True
from microscopes.common import validator
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
        # Save and validate model definition
        self._defn = defn
        validator.validate_len(data, defn.n, "data")

        # Validate kwargs
        valid_kwargs = ('dish_hps', 'vocab_hp',
                        'initial_dishes',
                        'topic_assignments',
                        'dish_assignments',
                        'table_assignments',)
        validator.validate_kwargs(kwargs, valid_kwargs)

        # Save and validate hyperparameters
        dish_hps = kwargs.get('dish_hps', None)
        if dish_hps is None:
            dish_hps = {'alpha': 0.1, 'gamma': 0.1}
        validator.validate_kwargs(dish_hps, ('alpha', 'gamma',))

        vocab_hp = kwargs.get('vocab_hp', 0.5)
        validator.validate_positive(vocab_hp)

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
    vocab_hp : parameter on symmetric Dirichlet prior over topic distributions (beta)
    dish_hps : concentration parameters on base (alpha) and second-level (gamma) Dirichlet processes
    """
    return state(defn=defn, data=data, r=r, **kwargs)