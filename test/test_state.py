from microscopes.common.rng import rng
from microscopes.lda.definition import model_definition
from microscopes.lda.model import initialize
from microscopes.lda.testutil import toy_dataset

from nose.tools import assert_equals


def test_simple():
    N, V = 10, 100
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = data
    prng = rng()
    s = initialize(defn, view, prng)
    assert_equals(s.nentities(), len(data))
