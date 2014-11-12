
from microscopes.common.rng import rng
from microscopes.lda.definition import model_definition
from microscopes.lda.model import (
    initialize,
    bind,
)
from microscopes.lda.testutil import toy_dataset
from microscopes.common.variadic.dataview import numpy_dataview

import numpy as np

from nose.tools import assert_equals


def test_simple():
    N, V = 10, 100
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = numpy_dataview(data)
    R = rng()
    s = initialize(defn, view, R)
    assert_equals(s.nentities(), len(data))


def test_explicit():
    N, V = 5, 100
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = numpy_dataview(data)
    R = rng()

    table_assignments = [
        np.random.randint(low=0, high=10, size=len(d)) for d in data]

    dish_assignments = [
        np.random.randint(low=0, high=len(t)) for t in table_assignments]

    s = initialize(defn, view, R,
        table_assignments=table_assignments,
        dish_assignments=dish_assignments)
    assert_equals(s.nentities(), len(data))
