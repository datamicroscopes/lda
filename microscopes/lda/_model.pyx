# cython: embedsignature=True


# python imports
from microscopes.common import validator
from microscopes.io.schema_pb2 import CRP


cdef class state:
    """The underlying state of an HDP-LDA

    You should not explicitly construct a state object.
    Instead, use :func:`initialize`.

    Notes
    -----
    This class is not meant to be sub-classed.

    """

    def __cinit__(self, model_definition defn, **kwargs):
        self._defn = defn
        cdef vector[vector[size_t]] c_dish_assignments
        cdef vector[size_t] c_dish_assignment

        # note: python cannot overload __cinit__(), so we
        # use kwargs to handle both the random initialization case and
        # the deserialize from string case
        if not (('data' in kwargs) ^ ('bytes' in kwargs)):
            raise ValueError("need exaclty one of `data' or `bytes'")

        valid_kwargs = ('data', 'bytes', 'r', 'initial_dishes',
                        'dish_hp', 'vocab_hp', 'dish_assignments',)
        validator.validate_kwargs(kwargs, valid_kwargs)

        if 'data' in kwargs:
            data = kwargs['data']
            validator.validate_type(data, abstract_dataview, "data")
            validator.validate_len(data, defn.n(), "data")

            if 'r' not in kwargs:
                raise ValueError("need parameter `r'")
            r = kwargs['r']
            validator.validate_type(r, rng, "r")

            def make_hp_bytes(hp):
                m = CRP()
                m.alpha = hp['alpha']
                return m.SerializeToString()

            dish_hp = kwargs.get('dish_hp', None)
            if dish_hp is None:
                dish_hp = {'alpha': 1.}
            validator.validate_type(dish_hp, dict, 'dish_hp')
            dish_hp_bytes = make_hp_bytes(dish_hp)

            vocab_hp = kwargs.get('vocab_hp', None)
            if vocab_hp is None:
                vocab_hp = {'alpha': 1.}
            validator.validate_type(vocab_hp, dict, 'vocab_hp')
            vocab_hp_bytes = make_hp_bytes(vocab_hp)

            initial_dishes = kwargs.get('initial_dishes', 10)
            validator.validate_positive(initial_dishes, 'initial_dishes')

            if 'dish_assignments' in kwargs:
                dish_assignments = list(kwargs['dish_assignments'])
                validator.validate_len(
                    dish_assignments, len(data), 'dish_assignments')
                for i, assignments in enumerate(dish_assignments):
                    validator.validate_len(assignments, data.rowsize(i))
                    c_dish_assignment.clear()
                    for dish in assignments:
                        validator.validate_in_range(dish, initial_dishes)
                        c_dish_assignment.push_back(dish)
                    c_dish_assignments.push_back(c_dish_assignment)

            self._thisptr = c_initialize(
                defn._thisptr.get()[0],
                dish_hp_bytes,
                vocab_hp_bytes,
                (<abstract_dataview> data)._thisptr.get()[0],
                initial_dishes,
                c_dish_assignments,
                (<rng> r)._thisptr[0])

            if self._thisptr.get() == NULL:
                raise RuntimeError("could not properly construct state")

    def nentities(self):
        return self._thisptr.get()[0].nentities()


cdef class document_model:
    def __cinit__(self, state s, abstract_dataview d):
        self._thisptr.reset(
            new c_document_model(
                s._thisptr, d._thisptr))


cdef class table_model:
    def __cinit__(self, state s, size_t document):
        validator.validate_in_range(document, s.nentities())
        self._thisptr.reset(
            new c_table_model(
                s._thisptr, document))


def bind(state s, **kwargs):
    valid_kwargs = ('data', 'document',)
    validator.validate_kwargs(kwargs, valid_kwargs)
    if 'data' in kwargs:
        return document_model(s, kwargs['data'])
    elif 'document' in kwargs:
        return table_model(s, kwargs['document'])
    else:
        raise ValueError("invalid arguments")


def initialize(model_definition defn,
               abstract_dataview data,
               rng r,
               **kwargs):
    """Initialize state to a random, valid point in the state space

    Parameters
    ----------
    defn : model definition
    data : variadic dataview
    rng : random state

    """
    return state(defn=defn, data=data, r=r, **kwargs)


def deserialize(model_definition defn, bytes):
    """Restore a state object from a bytestring representation.

    Note that a serialized representation of a state object does
    not contain its own structural definition.

    Parameters
    ----------
    defn : model definition
    bytes : bytestring representation

    """
    return state(defn=defn, bytes=bytes)


def _reconstruct_state(defn, bytes):
    return deserialize(defn, bytes)
