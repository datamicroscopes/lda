"""Utility functions
"""


def docs_from_document_term_matrix(dtm):
    """Read dataset from document term document-term matrix

    Parameters
    ----------
    dtm : array of shape N,V

    Returns
    -------
    docs: variadic array of N entites
    """
    docs = []
    for entity in dtm:
        doc = []
        for term_id, count in enumerate(entity):
            doc.extend(count * [term_id])
        docs.append(doc)
    return docs


def docs_from_ldac():
    """Read dataset from LDA-C formated file

    Parameters
    ----------
    dtm : array of shape N,V

    Returns
    -------
    docs: variadic array of N entites
    """
    pass
