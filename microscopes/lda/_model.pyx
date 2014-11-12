# cython: embedsignature=True


# python imports
from microscopes.common import validator
from microscopes.io.schema_pb2 import CRP
from distributions.io.schema_pb2 import DirichletDiscrete


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

        cdef vector[vector[size_t]] c_topic_assignments
        cdef vector[size_t] c_topic_assignment
        cdef vector[vector[size_t]] c_dish_assignments
        cdef vector[size_t] c_dish_assignment
        cdef vector[vector[size_t]] c_table_assignments
        cdef vector[size_t] c_table_assignment

        # note: python cannot overload __cinit__(), so we
        # use kwargs to handle both the random initialization case and
        # the deserialize from string case
        if not (('data' in kwargs) ^ ('bytes' in kwargs)):
            raise ValueError("need exaclty one of `data' or `bytes'")

        valid_kwargs = ('data', 'bytes', 'r',
                        'dish_hp',
                        'vocab_hps',
                        'initial_dishes',
                        'topic_assignments',
                        'dish_assignments',
                        'table_assignments',)
        validator.validate_kwargs(kwargs, valid_kwargs)

        if 'data' in kwargs:
            data = kwargs['data']
            validator.validate_type(data, abstract_dataview, "data")
            validator.validate_len(data, defn.n(), "data")

            if 'r' not in kwargs:
                raise ValueError("need parameter `r'")
            r = kwargs['r']
            validator.validate_type(r, rng, "r")

            def make_dish_hp_bytes(hp):
                m = CRP()
                m.alpha = hp['alpha']
                return m.SerializeToString()

            dish_hp = kwargs.get('dish_hp', None)
            if dish_hp is None:
                dish_hp = {'alpha': 1.}
            validator.validate_type(dish_hp, dict, 'dish_hp')
            dish_hp_bytes = make_dish_hp_bytes(dish_hp)

            def make_vocab_hps_bytes(hp):
                m = DirichletDiscrete.Shared()
                for alpha in hp['alphas']:
                    m.alphas.append(float(alpha))
                return m.SerializeToString()

            vocab_hps = kwargs.get('vocab_hps', None)
            if vocab_hps is None:
                vocab_hps = {'alphas': [1.]*defn.v()}
            validator.validate_len(vocab_hps['alphas'], defn.v())
            validator.validate_type(vocab_hps, dict, 'vocab_hps')
            vocab_hps_bytes = make_vocab_hps_bytes(vocab_hps)

            has_initial_dishes = 'initial_dishes' in kwargs
            has_topic_assignments = 'topic_assignments' in kwargs
            has_dish_assignments = 'dish_assignments' in kwargs
            has_table_assignments = 'table_assignments' in kwargs

            has_first = has_initial_dishes or has_topic_assignments
            has_second = has_dish_assignments or has_table_assignments

            if has_first and has_second:
                raise ValueError(
                    "cannot specify both `initial_dishes' and " +
                    "`dish_assignments' simultaneously, etc.")

            if has_first or (not has_first and not has_second):
                initial_dishes = kwargs.get('initial_dishes', 10)
                validator.validate_positive(initial_dishes, 'initial_dishes')

                if 'topic_assignments' in kwargs:
                    topic_assignments = list(kwargs['topic_assignments'])
                    validator.validate_len(
                        topic_assignments, len(data), 'topic_assignments')
                    for i, assignments in enumerate(topic_assignments):
                        validator.validate_len(assignments, data.rowsize(i))
                        c_topic_assignment.clear()
                        for topic in assignments:
                            validator.validate_in_range(topic, initial_dishes)
                            c_topic_assignment.push_back(topic)
                        c_topic_assignments.push_back(c_topic_assignment)

                self._thisptr = c_initialize(
                    defn._thisptr.get()[0],
                    dish_hp_bytes,
                    vocab_hps_bytes,
                    (<abstract_dataview> data)._thisptr.get()[0],
                    initial_dishes,
                    c_topic_assignments,
                    (<rng> r)._thisptr[0])

            else:
                if not has_dish_assignments or not has_table_assignments:
                    # XXX: implementation limitation
                    raise ValueError("need both dish/table assignments")

                dish_assignments = kwargs['dish_assignments']
                table_assignments = kwargs['table_assignments']

                validator.validate_len(
                    dish_assignments, len(data), 'dish_assignments')
                validator.validate_len(
                    table_assignments, len(data), 'table_assignments')

                # XXX: validate integrity of dish_assignments
                # XXX: validate integrity of table_assignments

                for assignments in dish_assignments:
                    c_dish_assignment.clear()
                    for dish in assignments:
                        c_dish_assignment.push_back(dish)
                    c_dish_assignments.push_back(c_dish_assignment)

                for i, assignments in enumerate(table_assignments):
                    validator.validate_len(assignments, data.rowsize(i))
                    c_table_assignment.clear()
                    for table in assignments:
                        c_table_assignment.push_back(table)
                    c_table_assignments.push_back(c_table_assignment)

                self._thisptr = c_initialize_explicit(
                    defn._thisptr.get()[0],
                    dish_hp_bytes,
                    vocab_hps_bytes,
                    (<abstract_dataview> data)._thisptr.get()[0],
                    c_dish_assignments,
                    c_table_assignments,
                    (<rng> r)._thisptr[0])

            if self._thisptr.get() == NULL:
                raise RuntimeError("could not properly construct state")

    def nentities(self):
        return self._thisptr.get()[0].nentities()

    def dish_assignments(self):
        return self._thisptr.get()[0].dish_assignments()

    def table_assignments(self):
        return self._thisptr.get()[0].table_assignments()

    def score_assignment(self):
        return self._thisptr.get()[0].score_assignment()

    def score_data(self, rng r):
        return self._thisptr.get()[0].score_data(r._thisptr[0])


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
