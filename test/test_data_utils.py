from microscopes.lda import utils


def test_docs_from_document_term_matrix_simple():
    docs = utils.docs_from_document_term_matrix([[3]])
    assert docs == [[0, 0, 0]]

    docs = utils.docs_from_document_term_matrix([[3, 2], [1, 1]])
    assert docs == [[0, 0, 0, 1, 1], [0, 1]]
