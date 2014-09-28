"""Implements the Runner interface fo LDA
"""

from microscopes.common import validator
from microscopes.common.rng import rng
from microscopes.common.variadic._dataview import abstract_dataview
from microscopes.lda.definition import model_definition
from microscopes.lda.model import state, bind
from microscopes.lda.kernels import assign, assign2

import itertools as it
import copy


def default_kernel_config(defn):
    """Creates a default kernel configuration suitable for general purpose
    inference.

    Parameters
    ----------
    defn : lda definition
    """
    # XXX(stephentu): fill me in
    return [('assign', {})]


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

    def __init__(self, defn, view, latent, kernel_config):
        validator.validate_type(defn, model_definition, 'defn')
        validator.validate_type(view, abstract_dataview, 'view')
        validator.validate_type(latent, state, 'latent')

        self._defn = defn
        self._view = view
        #self._latent = copy.deepcopy(latent)
        # XXX(stephentu): make copy work
        self._latent = latent

        self._kernel_config = []
        for kernel in kernel_config:
            name, config = kernel
            validator.validate_dict_like(config)
        if name == 'assign':
            pass
        else:
            raise ValueError("bad kernel found: {}".format(name))

        self._kernel_config.append((name, config))

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
        doc_model = bind(self._latent, data=self._view)
        for _ in xrange(niters):
            for name, config in self._kernel_config:
                if name == 'assign':
                    assign2(doc_model, r)
                    tabel_models = [
                        bind(self._latent, document=did)
                        for did in xrange(self._latent.nentities())
                    ]
                    for table_model in tabel_models:
                        assign(table_model, r)
                else:
                    assert False, 'should not be reached'
