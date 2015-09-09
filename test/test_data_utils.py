import numpy as np

from cStringIO import StringIO
from nose.tools import raises
from microscopes.lda import utils


def test_docs_from_document_term_matrix():
    dtm = [[2, 1], [3, 2]]
    docs = [[0, 0, 1], [0, 0, 0, 1, 1]]
    assert utils.docs_from_document_term_matrix(dtm) == docs


def test_docs_from_document_term_matrix_with_vocab():
    dtm = [[2, 1], [3, 2]]
    docs = [['cat', 'cat', 2], ['cat', 'cat', 'cat', 2, 2]]
    gen_docs = utils.docs_from_document_term_matrix(dtm, vocab=['cat', 2])
    assert gen_docs == docs


def test_docs_from_dtm_with_gaps():
    dtm = [[2, 0, 1], [1, 1, 1]]
    docs = [[0, 0, 2], [0, 1, 2]]
    assert utils.docs_from_document_term_matrix(dtm) == docs


def test_docs_from_numpy_dtp():
    dtm = np.array([[2, 1], [3, 2]])
    docs = [[0, 0, 1], [0, 0, 0, 1, 1]]
    assert utils.docs_from_document_term_matrix(dtm) == docs


def test_docs_from_ldac_simple():
    stream = StringIO()
    stream.write("2 0:2 1:1\n2 0:3 1:2")
    stream.seek(0) # rewind stream
    docs = [[0, 0, 1], [0, 0, 0, 1, 1]]
    assert utils.docs_from_ldac(stream) == docs

    stream = StringIO()
    stream.write("2 1:1 0:2\n3 2:1 0:3 1:1")
    stream.seek(0) # rewind stream
    docs = [[1, 0, 0], [2, 0, 0, 0, 1]]
    assert utils.docs_from_ldac(stream) == docs


@raises(AssertionError)
def test_bad_ldac_data():
    stream = StringIO()
    stream.write("2 0:1")
    stream.seek(0) # rewind stream
    utils.docs_from_ldac(stream)


def test_num_terms():
    docs = [[0, 1, 2], [1, 2, 3]]
    assert utils.num_terms(docs) == 4
