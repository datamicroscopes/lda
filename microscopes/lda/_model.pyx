# cython: embedsignature=True
import itertools
import numpy as np
import warnings

from microscopes.common import validator
from copy import deepcopy
from itertools import chain
from collections import Counter

\
def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning) #turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning) #reset filter
        return func(*args, **kwargs)
    return new_func


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

        self.vocab_hp = kwargs.get('vocab_hp', 0.5)
        validator.validate_positive(self.vocab_hp)

        # Get initial dishes or assigments
        dishes_and_tables = self._get_dishes_and_tables(kwargs)

        if 'initial_dishes' in dishes_and_tables:
            self._thisptr = c_initialize(
                defn=defn._thisptr.get()[0],
                alpha=dish_hps['alpha'],
                beta=self.vocab_hp,
                gamma=dish_hps['gamma'],
                initial_dishes=dishes_and_tables['initial_dishes'],
                docs=data,
                rng=r._thisptr[0])
        elif "table_assignments" in dishes_and_tables \
                and "dish_assignments" in dishes_and_tables:
            self._thisptr = c_initialize_explicit(
                defn=defn._thisptr.get()[0],
                alpha=dish_hps['alpha'],
                beta=self.vocab_hp,
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

    @deprecated
    def document_distribution(self):
        return self.topic_distribution_by_document()

    def topic_distribution_by_document(self):
        """Return list of distributions over topics for each document.

        Return value is a list of lists. Each entry in the outer list corresponds
        to a document. Each value in the each inner list is the probability of that
        topic being generated in this document.

        Commonly called Theta in the probablistic topic modeling literature.
        """
        doc_distribution = self._thisptr.get()[0].document_distribution()
        # Remove dummy topic
        distributions = [topic_distribution[1:] for topic_distribution in doc_distribution]
        # Normalize
        distributions = [[i/sum(d) for i in d] for d in distributions]
        return distributions

    @deprecated
    def word_distribution(self, rng r=None):
        return self.word_distribution_by_topic()

    def word_distribution_by_topic(self):
        """Return distribution over vocabulary words for each topic.

        Return value is a list of dictionaries. Each entry in the list corresponds
        to a topic. Each dictionary maps from the full vocabulary to the probability
        of the term being generated by that topic.

        Commoly called Phi in the probablistic topic modeling literature.
        """
        num_distribution_by_topic = self._thisptr.get()[0].word_distribution()
        # C++ returns a list of maps from integer representation of terms to
        # probabilities. Return the Python user a list of maps from the original
        # words (hashable objects) to probabilities.
        word_distribution_by_topic = []
        for num_dist in num_distribution_by_topic:
            word_dist = {}
            for num, prob in num_dist.iteritems():
                word = self._vocab[num]
                word_dist[word] = prob
            word_distribution_by_topic.append(word_dist)
        return word_distribution_by_topic

    def score_assignment(self):
        raise NotImplementedError()

    def score_data(self, rng r):
        raise NotImplementedError()

    def pyldavis_data(self, rng r=None):
        sorted_num_vocab = sorted(self._vocab.keys())

        topic_term_distribution = []
        for topic in self.word_distribution_by_topic():
            t = [topic[self._vocab[word_id]]
                 for word_id in sorted_num_vocab]
            topic_term_distribution.append(t)

        doc_topic_distribution = self.topic_distribution_by_document()

        doc_lengths = [len(doc) for doc in self._data]
        vocab = [self._vocab[k] for k in sorted_num_vocab]

        ctr = self._corpus_term_id_frequency()
        term_frequency = [ctr[num] for num in sorted_num_vocab]

        return {'topic_term_dists': topic_term_distribution,
                'doc_topic_dists': doc_topic_distribution,
                'doc_lengths': doc_lengths,
                'vocab': vocab,
                'term_frequency': term_frequency}

    def _corpus_term_id_frequency(self):
        flatten = lambda l: list(itertools.chain.from_iterable(l))
        ctr = Counter(flatten(self._data))
        return ctr

    def _corpus_term_frequency(self):
        flatten = lambda l: list(itertools.chain.from_iterable(l))
        flat_data = flatten(self._data)
        flat_vocab = [self._vocab[tid] for tid in flat_data]
        ctr = Counter(flat_vocab)
        return ctr

    def term_relevance_by_topic(self, weight=0.5):
        """For each topic, get terms sorted by relevance.

        Relevance metric is defined by Sievert and Shirley (2004).
        It is a weighted average of the log probability of a word
        occurring in a topic and the log lift of assigning the word
        to the topic.
        """
        phi = self.word_distribution_by_topic()
        ctf = self._corpus_term_frequency()
        relevance_by_topic = []
        for word_dist in phi:
            term_relevance = []
            for term, phi_kw in word_dist.iteritems():
                p_w = ctf[term]
                rel = self._relevance_for_word(phi_kw, p_w, weight)
                term_relevance.append((term, rel))
            term_relevance = sorted(term_relevance, reverse=True)
            relevance_by_topic.append(term_relevance)
        return relevance_by_topic

    def _relevance_for_word(self, phi_kw, p_w, weight=0.5):
        """

        Defined in LDAvis: A method for visualizing and interpreting topics (2014)
        by Sievert and Shirley
        """
        return weight * np.log(phi_kw) + \
            (1 - weight) * np.log(phi_kw / p_w)


    def predict(self, docs, prng, max_iter=20, tol=1e-16):
        """Predict topic distributions for documents
        Parameters
        ----------
        docs : document or list of documents
        max_iter : int, optional
            Maximum number of iterations.
        tol: double, optional
            Tolerance value used for stopping
        Returns
        -------
        list of topic distributions for each input document
        """
        if not isinstance(docs[0], list):
            docs = [docs]
        return [self._predict_single(doc, prng, max_iter, tol) for doc in docs]

    def _predict_single(self, doc, prng, max_iter, tol):
        PZS = np.zeros((len(doc), self.ntopics()))
        word_dist = self.word_distribution_by_topic()
        for iteration in range(max_iter + 1): # +1 is for initialization
            PZS_new = [[d[word] for d in word_dist]
                        for word in doc]
            PZS_new = np.array(PZS_new)
            PZS_new *= (PZS.sum(axis=0) - PZS + self.vocab_hp)
            PZS_new /= PZS_new.sum(axis=1)[:, np.newaxis] # vector to single column matrix
            PZS = PZS_new
            if np.abs(PZS_new - PZS).sum() < tol:
                break
        theta_doc = PZS.sum(axis=0) / PZS.sum()
        assert theta_doc.shape == (self.ntopics(),)
        return theta_doc.tolist()

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