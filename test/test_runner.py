from microscopes.lda import model, runner
from microscopes.lda.definition import model_definition
from microscopes.lda.testutil import toy_dataset, permutations
from microscopes.common.rng import rng
from microscopes.common.variadic.dataview import numpy_dataview
from microscopes.common.testutil import scores_to_probs

from nose.plugins.attrib import attr

import numpy as np


def test_runner_simple():
    N, V = 10, 100
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = numpy_dataview(data)
    prng = rng()
    latent = model.initialize(defn, view, prng)
    kc = runner.default_kernel_config(defn)
    r = runner.runner(defn, view, latent, kc)
    r.run(prng, 1)


@attr('wip')
def test_convergence_simple_1():
    N, V = 2, 10
    defn = model_definition(N, V)
    data = [
        np.array([5, 6]),
        np.array([0, 1, 2]),
    ]
    view = numpy_dataview(data)
    prng = rng()

    scores = []
    for tables, dishes in permutations([2, 3]):
        latent = model.initialize(
            defn, view, prng,
            table_assignments=tables,
            dish_assignments=dishes)
        scores.append(
            latent.score_assignment() +
            latent.score_data(prng))
    true_dist = scores_to_probs(scores)
