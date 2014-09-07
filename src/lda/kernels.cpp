#include <microscopes/lda/kernels.hpp>

namespace microscopes {
namespace kernels {

template void gibbsT::assign<lda::table_model>(lda::table_model &, common::rng_t &);
template void gibbsT::assign2<lda::document_model>(lda::document_model &, common::rng_t &);

} // namespace kernels
} // namespace microscopes
