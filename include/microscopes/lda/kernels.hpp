#pragma once

#include <microscopes/lda/model.hpp>

namespace microscopes {
namespace kernels {
namespace lda_crp {

extern std::vector<float>
calc_dish_posterior_t(microscopes::lda::state &state, size_t j, size_t t, common::rng_t &rng);

extern std::vector<float>
calc_dish_posterior_w(microscopes::lda::state &state, const std::vector<float> &f_k, common::rng_t &rng);

extern std::vector<float>
calc_f_k(microscopes::lda::state &state, size_t v, common::rng_t &rng);

extern std::vector<float>
calc_table_posterior(microscopes::lda::state &state, size_t j, std::vector<float> &f_k, common::rng_t &rng);

extern void
sampling_t(microscopes::lda::state &state, size_t j, size_t i, common::rng_t &rng);

extern void
sampling_k(microscopes::lda::state &state, size_t j, size_t t, common::rng_t &rng);
} // namespace lda_crp

extern void
lda_crp_gibbs(microscopes::lda::state &state, common::rng_t &rng);

} // namespace kernels
} // namespace microscopes