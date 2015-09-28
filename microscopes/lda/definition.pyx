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