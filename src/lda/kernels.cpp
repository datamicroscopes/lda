#include <microscopes/lda/kernels.hpp>

namespace microscopes {
namespace kernels {
namespace lda_crp {

std::vector<float>
calc_dish_posterior_t(microscopes::lda::state &state, size_t eid, size_t t, common::rng_t &rng) {
    std::vector<float> log_p_k(state.dishes_.size());

    auto k_old = state.dish_assignment(eid, t);
    auto n_jt_val = state.n_jt[eid][t];
    for (size_t i = 0; i < state.dishes_.size(); i++) {
        auto k = state.dishes_[i];
        float n_k_val = state.n_k.get(k); // V*beta when k == i == 0
        if (k == k_old) n_k_val -= n_jt_val;
        log_p_k[i] = distributions::fast_log(i == 0 ? state.gamma_ : state.m_k[k]);
        log_p_k[i] += distributions::fast_lgamma(n_k_val);
        log_p_k[i] -= distributions::fast_lgamma(n_k_val + n_jt_val);
    }

    for (auto &kv : state.n_jtv[eid][t]) {
        auto w = kv.first; // w is word index
        auto n_jtw = kv.second; // n_jtw is # of times word w appears at table t in doc eid.
        if (n_jtw == 0) continue; // if word w isn't at table t, continue. log_pk wouldn't change.

        for (size_t i = 0; i < state.dishes_.size(); i++) {
            float n_kw;
            n_kw = state.n_kv[state.dishes_[i]].get(w); // beta when k == i == 0
            if (state.dishes_[i] == state.dish_assignment(eid, t)) n_kw -= n_jtw;
            log_p_k[i] += distributions::fast_lgamma(n_kw + n_jtw);
            log_p_k[i] -= distributions::fast_lgamma(n_kw);
        }
    }

    std::vector<float> p_k;
    p_k.reserve(state.dishes_.size());
    float max_value = *std::max_element(log_p_k.begin(), log_p_k.end());
    for (auto log_p_k_value : log_p_k) {
        p_k.push_back(exp(log_p_k_value - max_value));
    }
    lda_util::normalize(p_k);
    return p_k;
}

std::vector<float>
calc_dish_posterior_w(microscopes::lda::state &state, const std::vector<float> &f_k, common::rng_t &rng){
    Eigen::VectorXf p_k(state.dishes_.size());
    for (size_t i = 0; i < state.dishes_.size(); ++i) {
        p_k(i) = state.m_k[state.dishes_[i]] * f_k[state.dishes_[i]];
    }
    p_k(0) = state.gamma_ / state.V;
    p_k /= p_k.sum();
    return std::vector<float>(p_k.data(), p_k.data() + p_k.size());
}

std::vector<float>
calc_f_k(microscopes::lda::state &state, size_t v, common::rng_t &rng) {
    Eigen::VectorXf f_k(state.n_kv.size());

    f_k(0) = 0;
    for (size_t k = 1; k < state.n_kv.size(); k++)
    {
        f_k(k) = state.n_kv[k].get(v) / state.n_k.get(k);
    }

    return std::vector<float>(f_k.data(), f_k.data() + f_k.size());
}

std::vector<float>
calc_table_posterior(microscopes::lda::state &state, size_t eid, std::vector<float> &f_k, common::rng_t &rng) {
    std::vector<size_t> using_table = state.using_t[eid];
    Eigen::VectorXf p_t(using_table.size());

    for (size_t i = 1; i < using_table.size(); i++) {
        auto p = using_table[i];
        p_t(i) = state.n_jt[eid][p] * f_k[state.dish_assignment(eid, p)];
    }
    Eigen::Map<Eigen::VectorXf> eigen_f_k(f_k.data(), f_k.size());
    Eigen::Map<Eigen::Matrix<size_t, Eigen::Dynamic, 1>> eigen_m_k(state.m_k.data(), state.m_k.size());
    float p_x_ji = state.gamma_ / state.V + eigen_f_k.dot(eigen_m_k.cast<float>());
    p_t(0) = p_x_ji * state.alpha_ / (state.gamma_ + state.ntables());
    p_t /= p_t.sum();
    return std::vector<float>(p_t.data(), p_t.data() + p_t.size());
}

void
sampling_t(microscopes::lda::state &state, size_t eid, size_t i, common::rng_t &rng) {
    state.remove_table(eid, i);
    size_t v = state.get_word(eid, i);
    std::vector<float> f_k = calc_f_k(state, v, rng);
    std::vector<float> p_t = calc_table_posterior(state, eid, f_k, rng);

    size_t t_new = state.using_t[eid][common::util::sample_discrete(p_t, rng)];
    if (t_new == 0)
    {
        auto p_k = calc_dish_posterior_w(state, f_k, rng);
        size_t k_new = state.dishes_[common::util::sample_discrete(p_k, rng)];
        if (k_new == 0) k_new = state.create_dish();
        t_new = state.create_table(eid, k_new);
    }
    state.add_table(eid, t_new, i);
}

void
sampling_k(microscopes::lda::state &state, size_t eid, size_t t, common::rng_t &rng) {
    state.leave_from_dish(eid, t);
    auto p_k = calc_dish_posterior_t(state, eid, t, rng);
    size_t k_new = state.dishes_[common::util::sample_discrete(p_k, rng)];
    if (k_new == 0) k_new = state.create_dish();
    state.seat_at_dish(eid, t, k_new);
}

} // namespace lda_crp

void
lda_crp_gibbs(microscopes::lda::state &state, common::rng_t &rng)
{
    for (size_t eid = 0; eid < state.nentities(); ++eid) {
        for (size_t i = 0; i < state.nterms(eid); ++i) {
            lda_crp::sampling_t(state, eid, i, rng);
        }
    }
    for (size_t eid = 0; eid < state.nentities(); ++eid) {
        for (auto t : state.using_t[eid]) {
            if (t != 0) {
                lda_crp::sampling_k(state, eid, t, rng);
            }
        }
    }
}

} // namespace kernels
} // namespace microscopes