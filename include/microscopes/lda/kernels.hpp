#pragma once

#include <microscopes/kernels/gibbs.hpp>
#include <microscopes/lda/model.hpp>

namespace microscopes {
namespace kernels {

extern template void gibbsT::assign<lda::table_model>(lda::table_model &, common::rng_t &);
extern template void gibbsT::assign2<lda::document_model>(lda::document_model &, common::rng_t &);

} // namespace kernels
} // namespace microscopes
