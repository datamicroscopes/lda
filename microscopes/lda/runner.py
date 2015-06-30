"""Implements the Runner interface fo LDA
"""

from microscopes.common import validator
from microscopes.common.rng import rng
from microscopes.common.variadic._dataview import abstract_dataview
from microscopes.lda.definition import model_definition
from microscopes.lda.model import state

import itertools as it
import copy


class runner(object):
    """The LDA runner

    Parameters
    ----------
    defn : ``model_definition``
        The structural definition.

    view : dataview
        The variadic dataview.

    latent : ``state``
        The initialization state.

    kernel_config : list

    """

    def __init__(self, defn, view, latent, kernel_config='assign'):
        self._defn = defn
        self._view = view
        self._latent = latent


    def run(self, r, niters=10000):
        """Run the specified mixturemodel kernel for `niters`, in a single
        thread.

        Parameters
        ----------
        r : random state
        niters : int

        """
        validator.validate_type(r, rng, param_name='r')
        validator.validate_positive(niters, param_name='niters')

        for _ in xrange(niters):
            print _
            # self._latent.inference()
