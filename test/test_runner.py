from microscopes.lda import model, runner
from microscopes.lda.definition import model_definition
from microscopes.lda.testutil import toy_dataset
from microscopes.common.rng import rng
from nose.tools import assert_almost_equals


def test_runner_simple():
    N, V = 10, 20
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = data
    prng = rng()
    latent = model.initialize(defn, view, prng)
    r = runner.runner(defn, view, latent)
    r.run(prng, 1)


def test_runner_specify_basic_kernel():
    N, V = 10, 20
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = data
    prng = rng()
    latent = model.initialize(defn, view, prng)
    r = runner.runner(defn, view, latent, ["crf"])
    r.run(prng, 1)


def test_runner_specify_hp_kernels():
    N, V = 10, 20
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    view = data
    prng = rng()
    latent = model.initialize(defn, view, prng)
    kernels = ['crf'] + \
        runner.second_dp_hp_kernel_config(defn) + \
        runner.base_dp_hp_kernel_config(defn)
    r = runner.runner(defn, view, latent, kernels)
    r.run(prng, 1)


def test_runner_base_dp_valid():
    N, V = 10, 20
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    prng = rng()
    latent = model.initialize(defn, data, prng)
    old_beta = latent.beta
    old_alpha = latent.alpha
    kernels = ['crf'] + \
        runner.base_dp_hp_kernel_config(defn)
    r = runner.runner(defn, data, latent, kernels)
    r.run(prng, 10)
    assert_almost_equals(latent.beta, old_beta)
    assert_almost_equals(latent.alpha, old_alpha)
    assert latent.gamma > 0


def test_runner_second_dp_valid():
    N, V = 10, 20
    defn = model_definition(N, V)
    data = toy_dataset(defn)
    prng = rng()
    latent = model.initialize(defn, data, prng)
    old_beta = latent.beta
    old_gamma = latent.gamma
    kernels = ['crf'] + \
        runner.second_dp_hp_kernel_config(defn)
    r = runner.runner(defn, data, latent, kernels)
    r.run(prng, 10)
    assert_almost_equals(latent.beta, old_beta)
    assert_almost_equals(latent.gamma, old_gamma)
    assert latent.alpha > 0
