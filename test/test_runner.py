from microscopes.lda.definition import model_definition
from microscopes.lda import model, runner
from microscopes.common.rng import rng


def test_runner_simple():
    defn = model_definition(n=10, v=20)
    r = rng()
    data = [[1, 2, 3, 4, 5], [2, 3, 4]]
    view = data
    latent = model.initialize(defn=defn, data=view, r=r)
    rnr = runner.runner(defn, view, latent)
    rnr.run(r=r, niters=10)
