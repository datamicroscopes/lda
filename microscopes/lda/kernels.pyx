# cython: embedsignature=True


# python imports
from microscopes.common import validator


def assign(table_model s, rng r):
    validator.validate_not_none(s, "s")
    validator.validate_not_none(r, "r")
    c_assign(s._thisptr.get()[0], r._thisptr[0])


def assign2(document_model s, rng r):
    validator.validate_not_none(s, "s")
    validator.validate_not_none(r, "r")
    c_assign2(s._thisptr.get()[0], r._thisptr[0])
