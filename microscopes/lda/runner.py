"""Implements the Runner interface fo LDA
"""

from microscopes.common import validator
from microscopes.common.rng import rng
from microscopes.lda.kernels import lda_crp_gibbs
from microscopes.lda.kernels import lda_sample_dispersion


class runner(object):
    """The LDA runner

    Parameters
    ----------
    defn : ``model_definition``
        The structural definition.

    view : dataview
        A list of list of serializable objects (the 'documents')

    latent : ``state``
        The initialization state.

    kernel_config : list

    """

    def __init__(self, defn, view, latent, kernel_config='assign'):
        self._defn = defn
        self._view = view
        self._latent = latent


    def run(self, r, niters=10000):
        """Run the specified lda kernel for `niters`, in a single
        thread.

        Parameters
        ----------
        r : random state
        niters : int

        """
        validator.validate_type(r, rng, param_name='r')
        validator.validate_positive(niters, param_name='niters')

        for _ in xrange(niters):
            lda_crp_gibbs(self._latent, r)
