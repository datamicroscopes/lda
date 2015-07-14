#include <microscopes/lda/model.hpp>


microscopes::lda::model_definition::model_definition(size_t n, size_t v)
    : n_(n), v_(v)
{
    MICROSCOPES_DCHECK(n > 0, "no docs");
    MICROSCOPES_DCHECK(v > 0, "no terms");
}

