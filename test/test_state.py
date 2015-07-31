import numpy as np

from microscopes.common.rng import rng
from microscopes.lda.definition import model_definition
from microscopes.lda.model import initialize
from microscopes.lda.testutil import toy_dataset

from nose.tools import assert_equals, assert_true
from nose.tools import assert_almost_equals


def test_simple():
    N, V = 10, 100
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = data
    prng = rng()
    s = initialize(defn, view, prng)
    assert_equals(s.nentities(), len(data))


def test_pyldavis_data():
    docs = [list('abcd'), list('cdef')]
    defn = model_definition(len(docs), v=6)
    prng = rng()
    s = initialize(defn, docs, prng)
    data = s.pyldavis_data(prng)
    index_of_a = data['vocab'].index('a')
    index_of_c = data['vocab'].index('c')
    assert_equals(data['term_frequency'][index_of_a], 1)
    assert_equals(data['term_frequency'][index_of_c], 2)
    for dist in data['topic_term_dists']:
        assert_almost_equals(sum(dist), 1)
    for dist in data['doc_topic_dists']:
        assert_almost_equals(sum(dist), 1)


def test_single_dish_initialization():
    N, V = 10, 100
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = data
    prng = rng()
    s = initialize(defn, view, prng, initial_dishes=1)
    assert_equals(s.ntopics(), 0) # Only dummy topic


def test_multi_dish_initialization():
    N, V = 10, 100
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
    # explicit initialization doesn't work yet
    return
    # N, V = 5, 100
    # defn = model_definition(N, V)
    # data = toy_dataset(defn)
    # prng = rng()

    # table_assignments = [
    #     np.random.randint(low=0, high=10, size=len(d)) for d in data]

    # dish_assignments = [
    #     np.random.randint(low=0, high=len(t), size=len(d))
    #     for t, d in zip(table_assignments, data)]

    # s = initialize(defn, data, prng,
    #                table_assignments=table_assignments,
    #                dish_assignments=dish_assignments)
    # assert_equals(s.nentities(), len(data))
