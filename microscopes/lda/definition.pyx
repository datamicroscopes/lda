# cython: embedsignature=True

cdef class model_definition:
    def __cinit__(self, n, v):
        """The LDA model structural definition

        Parameters
        ----------
        n : Number of documents in the model
        v :  Number of unique vocabulary words in the model
        """
        self.n = n
        self.v = v
        self._thisptr.reset(new c_model_definition(n, v))

    def __reduce__(self):
        args = (self.n, self.v,)
        return (_reconstruct_model_definition, args)


def _reconstruct_model_definition(n, v):
    return model_definition(n, v)