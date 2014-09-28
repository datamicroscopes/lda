from microscopes.lda._kernels_h cimport (
    assign as c_assign,
    assign2 as c_assign2,
)
from microscopes.lda._model cimport (
    table_model,
    document_model,
)
from microscopes.common._rng cimport rng
