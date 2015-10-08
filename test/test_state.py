import itertools
import pickle
import cPickle

from microscopes.common.rng import rng
from microscopes.lda.definition import model_definition
from microscopes.lda.model import initialize, deserialize
from microscopes.lda.testutil import toy_dataset

from nose.tools import assert_equals, assert_true, raises
from nose.tools import assert_almost_equals, assert_raises


def vocab_size(docs):
    flatten = lambda l: list(itertools.chain.from_iterable(l))
    return len(set(flatten(docs)))


def test_simple():
    N, V = 10, 20
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = data
    prng = rng()
    s = initialize(defn, view, prng)
    assert_equals(s.nentities(), len(data))


def test_serialize_simple():
    docs = [list('abcd'), list('cdef')]
    defn = model_definition(len(docs), v=6)
    prng = rng()
    s = initialize(defn, docs, prng)
    m = s.serialize()
    s2 = deserialize(defn, m)
    assert s2.__class__ == s.__class__
    assert all(word in "abcdef"
               for wd in s2.word_distribution_by_topic()
               for word in wd.keys())
    assert all(isinstance(word, str)
               for wd in s2.word_distribution_by_topic()
               for word in wd.keys())


def test_serialize_pickle():
    docs = [list('abcd'), list('cdef')]
    defn = model_definition(len(docs), v=6)
    prng = rng()
    s = initialize(defn, docs, prng)
    # Pickle
    bstr = pickle.dumps(s)
    s2 = pickle.loads(bstr)
    assert s2.__class__ == s.__class__

    # cPickle
    bstr = cPickle.dumps(s)
    s2 = cPickle.loads(bstr)
    assert s2.__class__ == s.__class__


@raises(ValueError)
def test_cant_serialize():
    N, V = 10, 20
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    prng = rng()
    s = initialize(defn, data, prng)
    s.serialize()


def test_pyldavis_data():
    docs = [list('abcd'), list('cdef')]
    defn = model_definition(len(docs), v=6)
    prng = rng()
    s = initialize(defn, docs, prng)
    data = s.pyldavis_data()
    index_of_a = data['vocab'].index('a')
    index_of_c = data['vocab'].index('c')
    assert_equals(data['term_frequency'][index_of_a], 1)
    assert_equals(data['term_frequency'][index_of_c], 2)
    for dist in data['topic_term_dists']:
        assert_almost_equals(sum(dist), 1)
    for dist in data['doc_topic_dists']:
        assert_almost_equals(sum(dist), 1)


def test_relevance():
    docs = [list('abcd'), list('cdef')]
    defn = model_definition(len(docs), v=6)
    prng = rng()
    s = initialize(defn, docs, prng)
    s.term_relevance_by_topic(weight=0)
    s.term_relevance_by_topic(weight=1)
    rel = s.term_relevance_by_topic()
    assert isinstance(rel, list)
    assert isinstance(rel[0], list)
    assert len(rel) == s.ntopics()
    assert len(rel[0]) == s.nwords()
    assert rel[0] == sorted(rel[0],
                            key=lambda (_, r): r,
                            reverse=True)
    assert rel[-1][0] < rel[-1][-1]


def test_single_dish_initialization():
    N, V = 10, 20
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = data
    prng = rng()
    s = initialize(defn, view, prng, initial_dishes=1)
    assert_equals(s.ntopics(), 0) # Only dummy topic


def test_multi_dish_initialization():
    N, V = 10, 20
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = data
    prng = rng()
    s = initialize(defn, view, prng, initial_dishes=V)
    assert_true(s.ntopics() > 1)


def test_alpha_numeric():
    docs = [list('abcd'), list('cdef')]
    defn = model_definition(len(docs), v=6)
    prng = rng()
    s = initialize(defn, docs, prng)
    assert_equals(s.nentities(), len(docs))
    assert_equals(s.nwords(), 6)


def test_explicit():
    """Test that we can explicitly initialize state by specifying
    table and dish assignments
    """
    prng = rng()
    N, V = 3, 7
    defn = model_definition(N, V)
    data = [[0, 1, 2, 3], [0, 1, 4], [0, 1, 5, 6]]

    table_assignments = [[1, 2, 1, 2], [1, 1, 1], [3, 3, 3, 1]]
    dish_assignments = [[0, 1, 2], [0, 3], [0, 1, 2, 1]]

    s = initialize(defn, data,
                   table_assignments=table_assignments,
                   dish_assignments=dish_assignments)
    assert_equals(s.nentities(), len(data))
    assert len(s.dish_assignments()) == len(dish_assignments)
    assert len(s.table_assignments()) == len(table_assignments)
    for da1, da2 in zip(s.dish_assignments(), dish_assignments):
        assert da1 == da2
    for ta1, ta2 in zip(s.table_assignments(), table_assignments):
        assert ta1 == ta2


def test_explicit_exceptions():
    """ValueError should be rasied for bad assignments
    """
    prng = rng()
    N, V = 3, 7
    defn = model_definition(N, V)
    data = [[0, 1, 2, 3], [0, 1, 4], [0, 1, 5, 6]]

    # We should get an error if we leave out a dish assignment for a given table
    table_assignments = [[1, 2, 1, 2], [1, 1, 1], [3, 3, 3, 1]]
    dish_assignments = [[0, 1, 2], [0, 3], [0, 1, 2]]

    assert_raises(ValueError,
                  initialize,
                  defn, data,
                  table_assignments=table_assignments,
                  dish_assignments=dish_assignments)

    # We should get an error if we leave out a table assignment for a given word
    table_assignments = [[1, 2, 1, 2], [1, 1, 1], [3, 3, 3]]
    dish_assignments = [[0, 1, 2], [0, 3], [0, 1, 2, 1]]

    assert_raises(ValueError,
                  initialize,
                  defn, data,
                  table_assignments=table_assignments,
                  dish_assignments=dish_assignments)


def test_explicit_inception():
    """Initialize a new state using assignments from old

    Helps ensure that our assignment validation code is correct
    """
    prng = rng()
    N, V = 3, 7
    defn = model_definition(N, V)
    data = [[0, 1, 2, 3], [0, 1, 4], [0, 1, 5, 6]]

    table_assignments = [[1, 2, 1, 2], [1, 1, 1], [3, 3, 3, 1]]
    dish_assignments = [[0, 1, 2], [0, 3], [0, 1, 2, 1]]

    s = initialize(defn, data,
                   table_assignments=table_assignments,
                   dish_assignments=dish_assignments)
    s2 = initialize(defn, data,
                    table_assignments=s.table_assignments(),
                    dish_assignments=s.dish_assignments())
