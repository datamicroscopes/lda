# cython: embedsignature=True
from microscopes.common import validator
from copy import deepcopy
from itertools import chain

cdef class state:
    """The underlying state of an HDP-LDA
    You should not explicitly construct a state object.
    Instead, use :func:`initialize`.
    Notes
    -----
    This class is not meant to be sub-classed.
    """
    def __cinit__(self, model_definition defn,
                  vector[vector[size_t]] data,
                  vocab,
                  rng r, **kwargs):
        # Save and validate model definition
        self._defn = defn
        self._vocab = vocab
        validator.validate_len(data, defn.n, "data")

        for doc in data:
            for word in doc:
                if word >= defn.v:
                    raise ValueError("Word index out of bounds.")

        # Validate kwargs
        valid_kwargs = ('dish_hps', 'vocab_hp',
                        'initial_dishes',
                        'initial_tables',
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

        # Get initial dishes or assigments
        dishes_and_tables = self._get_dishes_and_tables(kwargs)

        if 'initial_dishes' in dishes_and_tables:
            self._thisptr = c_initialize(
                defn=defn._thisptr.get()[0],
                alpha=dish_hps['alpha'],
                beta=vocab_hp,
                gamma=dish_hps['gamma'],
                initial_dishes=dishes_and_tables['initial_dishes'],
                docs=data,
                rng=r._thisptr[0])
        elif "table_assignments" in dishes_and_tables \
                and "dish_assignments" in dishes_and_tables:
            self._thisptr = c_initialize_explicit(
                defn=defn._thisptr.get()[0],
                alpha=dish_hps['alpha'],
                beta=vocab_hp,
                gamma=dish_hps['gamma'],
                dish_assignments=dishes_and_tables['dish_assignments'],
                table_assignments=dishes_and_tables['table_assignments'],
                docs=data,
                rng=r._thisptr[0])
        else:
            raise NotImplementedError(("Specify either: (1) initial_dishes, "
                "(2) table_assignments and dish_assignments, or (3) none of the above."))

    DEFAULT_INITIAL_DISH_HINT = 10

    def perplexity(self):
        return self._thisptr.get().perplexity()

    def nentities(self):
        return self._thisptr.get().nentities()

    def ntopics(self):
        return self._thisptr.get().ntopics()

    def nwords(self):
        return self._thisptr.get().nwords()

    def assignments(self):
        return self._thisptr.get()[0].assignments()

    def dish_assignments(self):
        return self._thisptr.get()[0].dish_assignments()

    def table_assignments(self):
        return self._thisptr.get()[0].table_assignments()

    def document_distribution(self):
        doc_distribution = self._thisptr.get()[0].document_distribution()
        # Remove dummy topic
        distributions = [topic_distribution[1:] for topic_distribution in doc_distribution]
        distributions = [[i/sum(d) for i in d] for d in distributions]
        return distributions

    def word_distribution(self, rng r):
        return self._thisptr.get()[0].word_distribution()

    def score_assignment(self):
        raise NotImplementedError()
        return self._thisptr.get()[0].score_assignment()

    def score_data(self, rng r):
        raise NotImplementedError()
        return self._thisptr.get()[0].score_data(r._thisptr[0])

    def _get_dishes_and_tables(self, kwargs):
        if "initial_dishes" in kwargs \
                and "table_assignments" not in kwargs \
                and "dish_assignments" not in kwargs:
            return {'initial_dishes': kwargs["initial_dishes"]}

        elif "table_assignments" in kwargs \
                and "dish_assignments" in kwargs \
                and "initial_dishes" not in kwargs:
            return {'table_assignments': kwargs['table_assignments'],
                    'dish_assignments': kwargs['dish_assignments']}

        else:
            return {'initial_dishes': self.DEFAULT_INITIAL_DISH_HINT}


def bind(state s, **kwargs):
    pass

def initialize(model_definition defn, data, rng r, **kwargs):
    """Initialize state to a random, valid point in the state space
    Parameters
    ----------
    defn : model definition
    data : a list of list of serializable objects (i.e. 'documents')
    rng : random state
    vocab_hp : parameter on symmetric Dirichlet prior over topic distributions (beta)
    dish_hps : concentration parameters on base (alpha) and second-level (gamma) Dirichlet processes
    """
    numeric_docs, vocab_lookup = _initialize_data(data)
    return state(defn=defn, data=numeric_docs, vocab=vocab_lookup, r=r, **kwargs)

def _initialize_data(docs):
    """Convert docs (list of list of hashable items) to list of list of
    positive integers and a map from the integers back to the terms
    """
    vocab = set(chain.from_iterable(docs))
    word_to_int = { word: i for i, word in enumerate(vocab)}
    int_to_word = { i: word for i, word in enumerate(vocab)}
    print int_to_word
    numeric_docs = []
    for doc in docs:
        numeric_docs.append([word_to_int[word] for word in doc])
    return numeric_docs, int_to_word