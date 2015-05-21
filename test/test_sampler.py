from nose.plugins.attrib import attr

from microscopes.lda import model, kernels
from microscopes.lda.definition import model_definition
from microscopes.lda.testutil import permutations

from microscopes.common.rng import rng
from microscopes.common.variadic.dataview import numpy_dataview
from microscopes.common.testutil import (
    assert_discrete_dist_approx,
    permutation_canonical,
    scores_to_probs,
)


import numpy as np


@attr('slow')
def test_convergence_simple():
    N, V = 2, 10
    defn = model_definition(N, V)
    data = [
        np.array([5, 6]),
        np.array([0, 1, 2]),
    ]
    view = numpy_dataview(data)
    prng = rng()

    scores = []
    idmap = {}
    for i, (tables, dishes) in enumerate(permutations([2, 3])):
        latent = model.initialize(
            defn, view, prng,
            table_assignments=tables,
            dish_assignments=dishes)
        scores.append(
            latent.score_assignment() +
            latent.score_data(prng))
        idmap[(tables, dishes)] = i
    true_dist = scores_to_probs(scores)

    def kernel(latent):
        # mutates latent in place
        doc_model = model.bind(latent, data=view)
        kernels.assign2(doc_model, prng)
        for did in xrange(latent.nentities()):
            table_model = model.bind(latent, document=did)
            kernels.assign(table_model, prng)

    latent = model.initialize(defn, view, prng)

    skip = 10

    def sample_fn():
        for _ in xrange(skip):
            kernel(latent)
        table_assignments = latent.table_assignments()
        canon_table_assigments = tuple(
            map(tuple, map(permutation_canonical, table_assignments)))

        dish_maps = latent.dish_assignments()
        dish_assignments = []
        for dm, (ta, ca) in zip(dish_maps, zip(table_assignments, canon_table_assigments)):
            dish_assignment = []
            for t, c in zip(ta, ca):
                if c == len(dish_assignment):
                    dish_assignment.append(dm[t])
            dish_assignments.append(dish_assignment)

        canon_dish_assigments = tuple(
            map(tuple, map(permutation_canonical, dish_assignments)))

        return idmap[(canon_table_assigments, canon_dish_assigments)]

    assert_discrete_dist_approx(
        sample_fn, true_dist,
        ntries=100, nsamples=10000, kl_places=2)
