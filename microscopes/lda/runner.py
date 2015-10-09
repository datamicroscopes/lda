"""Implements the Runner interface fo LDA
"""

from microscopes.common import validator
from microscopes.common.rng import rng
from microscopes.lda.definition import model_definition
from microscopes.lda.kernels import lda_crp_gibbs
from microscopes.lda.kernels import sample_gamma, sample_alpha, sample_beta


def _validate_definition(defn):
    if not isinstance(defn, model_definition):
        raise ValueError("bad defn given")
    return defn


def crf_kernel_config(defn):
    """Creates a default kernel configuration for sampling the dish assignment
    using the "Posterior sampling in the Chinese restaurant franchise" Gibbs
    sampler from Teh et al (2005)

    Parameters
    ----------
    defn : LDA model definition
    """
    return ['crf']


def base_dp_hp_kernel_config(defn, hp1=5, hp2=.1):
    """Sample the base level Dirichlet process parameter (gamma)
    using the method of Escobar and West (1995) with n = T.

    Parameters
    ----------
    defn : LDA model definition
    """
    return [('direct_base_dp_hp', {'hp1': hp1, 'hp2': hp2})]


def second_dp_hp_kernel_config(defn, hp1=5, hp2=0.1):
    """Sample the second level Dirichlet process parameter (alpha)
    using the method of Teh et al (2005).

    Current implementation is based on that of Gregor Heinrich
    available at http://bit.ly/1LkdBdX.

    Teh (2005) is available here:
        http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf
    Heinrich says his method is based on equations 47-49
    ----------
    defn : LDA model definition
    """
    return [('direct_second_dp_hp', {'hp1': hp1, 'hp2': hp2})]


class runner(object):
    """The LDA runner

    Parameters
    ----------
    defn : ``model_definition``: The structural definition.
    view :  A list of list of serializable objects (the 'documents')
    latent : ``state``: The initialization state.
    kernel_config : list
        A list of either `x` strings or `(x, y)` tuples, where `x` is a string
        containing the name of the kernel and `y` is a dict which configures
        the particular kernel. In the former case where `y` is omitted, then
        the defaults parameters for each kernel are used.
        Possible values of `x` are:
        {'crp', 'direct_base_dp_hp', 'direct_second_dp_hp', 'direct_vocab_hp'}
    """

    def __init__(self, defn, view, latent, kernel_config=('crf', )):
        self._defn = defn
        self._view = view
        self._latent = latent
        self._kernel_config = []

        for kernel in kernel_config:
            if hasattr(kernel, '__iter__'):
                name, config = kernel
            else:
                name, config = kernel, {}
            validator.validate_dict_like(config)
            self._kernel_config.append((name, config))


    def run(self, r, niters=10000):
        """Run the lda kernel for `niters`, in a single thread.

        Parameters
        ----------
        r : random state
        niters : int

        """
        validator.validate_type(r, rng, param_name='r')
        validator.validate_positive(niters, param_name='niters')

        for _ in xrange(niters):
            for name, config in self._kernel_config:
                if name == 'crf':
                    lda_crp_gibbs(self._latent, r)
                elif name == 'direct_base_dp_hp':
                    print "sample gamma"
                    sample_gamma(self._latent, r, config['hp1'], config['hp2'])
                elif name == 'direct_second_dp_hp':
                    print "sample alpha"
                    sample_alpha(self._latent, r, config['hp1'], config['hp2'])
                    print self._latent.alpha
                else:
                    assert False, "should not be reach"
