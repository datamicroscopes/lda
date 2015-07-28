import numpy as np

from microscopes.common.rng import rng
from microscopes.lda.definition import model_definition
from microscopes.lda.model import initialize
from microscopes.lda.testutil import toy_dataset

from nose.tools import assert_equals, assert_true


def test_simple():
    N, V = 10, 100
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = data
    prng = rng()
    s = initialize(defn, view, prng)
    assert_equals(s.nentities(), len(data))


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
