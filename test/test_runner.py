from microscopes.lda import model, runner
from microscopes.lda.definition import model_definition
from microscopes.lda.testutil import toy_dataset
from microscopes.common.rng import rng


def test_runner_simple():
    N, V = 10, 20
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = data
    prng = rng()
    latent = model.initialize(defn, view, prng)
    r = runner.runner(defn, view, latent)
    r.run(prng, 1)
