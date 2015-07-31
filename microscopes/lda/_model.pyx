# cython: embedsignature=True
import itertools

from microscopes.common import validator
from copy import deepcopy
from itertools import chain
from collections import Counter

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
        self._data = data
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
        # Normalize
        distributions = [[i/sum(d) for i in d] for d in distributions]
        return distributions

    def word_distribution(self, rng r):
        num_distribution_by_topic = self._thisptr.get()[0].word_distribution()
        # C++ returns a list of maps from integer representation of terms to
        # probabilities. Return the Python user a list of maps from the original
        # words (hashable objects) to probabilities.
        word_distribution_by_topic = []
        for num_dist in word_distribution_by_topic:
            word_dist = {}
            for num, prob in num_dist.items():
                word_dist[self.vocab[num]] = num_dist[num]
        return word_distribution_by_topic

    def score_assignment(self):
        raise NotImplementedError()
        return self._thisptr.get()[0].score_assignment()

    def score_data(self, rng r):
        raise NotImplementedError()
        return self._thisptr.get()[0].score_data(r._thisptr[0])

    def pyldavis_data(self, rng r):
        sorted_num_vocab = sorted(self._vocab.keys())

        topic_term_distribution = []
        for topic in self.word_distribution(r):
            t = [topic[word_id] for word_id in sorted_num_vocab]
            topic_term_distribution.append(t)

        doc_topic_distribution = self.document_distribution()

        doc_lengths = [len(doc) for doc in self._data]
        vocab = [self._vocab[k] for k in sorted_num_vocab]

        flatten = lambda l: list(itertools.chain.from_iterable(l))
        ctr = Counter(flatten(self._data))
        term_frequency = [ctr[num] for num in sorted_num_vocab]

        return {'topic_term_dists': topic_term_distribution,
                'doc_topic_dists': doc_topic_distribution,
                'doc_lengths': doc_lengths,
                'vocab': vocab,
                'term_frequency': term_frequency}

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
    validator.validate_len(vocab_lookup, defn.v, "vocab_lookup")
    if len(vocab_lookup) == defn.v
    return state(defn=defn, data=numeric_docs, vocab=vocab_lookup, r=r, **kwargs)

def _initialize_data(docs):
    """Convert docs (list of list of hashable items) to list of list of
    positive integers and a map from the integers back to the terms
    """
    vocab = set(chain.from_iterable(docs))
    word_to_int = { word: i for i, word in enumerate(vocab)}
    int_to_word = { i: word for i, word in enumerate(vocab)}
    numeric_docs = []
    for doc in docs:
        numeric_docs.append([word_to_int[word] for word in doc])
    return numeric_docs, int_to_word