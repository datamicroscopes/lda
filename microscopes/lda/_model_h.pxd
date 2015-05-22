from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.set cimport set
from libcpp cimport bool as cbool
from libc.stddef cimport size_t

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.common.recarray._dataview_h cimport row_accessor, row_mutator, dataview
from microscopes.common._random_fwd_h cimport rng_t
from microscopes.common._typedefs_h cimport hyperparam_bag_t, suffstats_bag_t
from microscopes.common._runtime_type_h cimport runtime_type
from microscopes.common._entity_state_h cimport entity_based_state_object
from microscopes._models_h cimport model as c_model

cdef extern from "microscopes/lda/model.hpp" namespace "microscopes::lda":

    cdef cppclass model_definition:
        pass
        model_definition(size_t, const vector[shared_ptr[c_model]] &) except +
        vector[runtime_type] get_runtime_types() except +
        size_t nmodels()


    cdef cppclass state:
        int temp() except +