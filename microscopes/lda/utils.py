"""Utility functions
"""


def num_terms(docs):
    terms = set()
    for doc in docs:
        terms.update(set(doc))
    return len(terms)


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
    for term_counts in dtm:
        term_counts = enumerate(term_counts)
        docs.append(_term_counts_to_doc(term_counts))
    return docs


def docs_from_ldac(stream):
    """Read dataset from LDA-C formated file

    From David Blei:

    Under LDA, the words of each document are assumed exchangeable.  Thus,
    each document is succinctly represented as a sparse vector of word
    counts. The data is a file where each line is of the form:

         [M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]

    where [M] is the number of unique terms in the document, and the
    [count] associated with each term is how many times that term appeared
    in the document.  Note that [term_1] is an integer which indexes the
    term; it is not a string.

    source: http://www.cs.princeton.edu/~blei/lda-c/readme.txt

    Parameters
    ----------
    stream: file object
        File yielding unicode strings in LDA-C format.


    Returns
    -------
    docs: variadic array of N entites
    """
    n_entities = 0
    docs = []
    for line in stream:
        line = line.strip().split(' ')
        if len(line) == 1 and line[0] == '':
            continue
        unique_terms = int(line.pop(0))
        term_counts = [tc.split(":") for tc in line]
        term_counts = [(int(t), int(c)) for t, c in term_counts]
        assert unique_terms == len(term_counts)
        docs.append(_term_counts_to_doc(term_counts))
        n_entities += 1
    return docs


def _term_counts_to_doc(term_counts):
    doc = []
    for term_id, count in term_counts:
            doc.extend(count * [term_id])
    return doc
