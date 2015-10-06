"""Implements the Runner interface fo LDA
"""

from microscopes.common import validator
from microscopes.common.rng import rng
from microscopes.lda.kernels import lda_crp_gibbs
from microscopes.lda.kernels import sample_gamma, sample_alpha, sample_beta


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
                elif name == 'direct_second_dp_hp':
                    sample_gamma(self._latent, r, config['a'], config['b'])
                elif name == 'direct_base_dp_hp':
                    sample_alpha(self._latent, r, config['a'], config['b'])
                elif name == 'direct_vocab_hp':
                    sample_beta(self._latent, r, config['a'], config['b'])
                else:
                    assert False, "should not be reach"
            lda_crp_gibbs(self._latent, r)
            sample_gamma(self._latent, r, 5, 0.1)
            sample_alpha(self._latent, r, 5, 0.1)
