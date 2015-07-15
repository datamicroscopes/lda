#pragma once

#include <microscopes/lda/model.hpp>


namespace microscopes {
namespace kernels {
namespace lda_crp {
std::vector<float>
calc_dish_posterior_t(microscopes::lda::state &state, size_t j, size_t t, common::rng_t &rng) {
    std::vector<float> log_p_k(state.dishes_.size());

    auto k_old = state.restaurants_[j][t];
    auto n_jt_val = state.n_jt[j][t];
    for (size_t i = 0; i < state.dishes_.size(); i++) {
        auto k = state.dishes_[i];
        if (k == 0) continue;
        float n_k_val = (k == k_old) ? state.n_k.get(k) - state.n_jt[j][t] : state.n_k.get(k);
        assert(n_k_val > 0);
        log_p_k[i] = distributions::fast_log(state.m_k[k]) + distributions::fast_lgamma(n_k_val) - distributions::fast_lgamma(n_k_val + n_jt_val);
        assert(isfinite(log_p_k[i]));
    }
    log_p_k[0] = distributions::fast_log(state.gamma_) + distributions::fast_lgamma(state.V * state.beta_) - distributions::fast_lgamma(state.V * state.beta_ + state.n_jt[j][t]);

    for (auto &kv : state.n_jtv[j][t]) {
        auto w = kv.first;
        auto n_jtw = kv.second;
        if (n_jtw == 0) continue;
        assert(n_jtw > 0);

        std::vector<float> n_kw(state.dishes_.size());
        for (size_t i = 0; i < state.dishes_.size(); i++) {
            n_kw[i] = state.n_kv[state.dishes_[i]].get(w);
            if (state.dishes_[i] == state.restaurants_[j][t]) n_kw[i] -= n_jtw;
            assert(i == 0 || n_kw[i] > 0);
        }
        n_kw[0] = 1; // # dummy for logarithm's warning
        for (size_t i = 1; i < n_kw.size(); i++) {
            log_p_k[i] += distributions::fast_lgamma(n_kw[i] + n_jtw) - distributions::fast_lgamma(n_kw[i]);
        }
        log_p_k[0] += distributions::fast_lgamma(state.beta_ + n_jtw) - distributions::fast_lgamma(state.beta_);
    }
    for (auto x : log_p_k) assert(isfinite(x));

    std::vector<float> p_k;
    p_k.reserve(state.dishes_.size());
    float max_value = *std::max_element(log_p_k.begin(), log_p_k.end());
    for (auto log_p_k_value : log_p_k) {
        p_k.push_back(exp(log_p_k_value - max_value));
    }
    util::normalize(p_k);
    return p_k;
}

std::vector<float>
calc_dish_posterior_w(microscopes::lda::state &state, const std::vector<float> &f_k, common::rng_t &rng) {
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

    f_k(0) = (state.n_kv[0].get(v) - state.beta_) / state.n_k.get(0);
    for (size_t k = 1; k < state.n_kv.size(); k++)
    {
        f_k(k) = state.n_kv[k].get(v) / state.n_k.get(k);
    }

    return std::vector<float>(f_k.data(), f_k.data() + f_k.size());
}

std::vector<float>
calc_table_posterior(microscopes::lda::state &state, size_t j, std::vector<float> &f_k, common::rng_t &rng) {
    std::vector<size_t> using_table = state.using_t[j];
    Eigen::VectorXf p_t(using_table.size());

    for (size_t i = 0; i < using_table.size(); i++) {
        auto p = using_table[i];
        p_t(i) = state.n_jt[j][p] * f_k[state.restaurants_[j][p]];
    }
    Eigen::Map<Eigen::VectorXf> eigen_f_k(f_k.data(), f_k.size());
    Eigen::Map<Eigen::Matrix<size_t, Eigen::Dynamic, 1>> eigen_m_k(state.m_k.data(), state.m_k.size());
    float p_x_ji = state.gamma_ / state.V + eigen_f_k.dot(eigen_m_k.cast<float>());
    p_t[0] = p_x_ji * state.alpha_ / (state.gamma_ + state.m);
    p_t /= p_t.sum();
    return std::vector<float>(p_t.data(), p_t.data() + p_t.size());
}

void
sampling_t(microscopes::lda::state &state, size_t j, size_t i, common::rng_t &rng) {
    state.remove_table(j, i);
    size_t v = state.x_ji[j][i];
    std::vector<float> f_k = calc_f_k(state, v, rng);
    assert(f_k[0] == 0);
    std::vector<float> p_t = calc_table_posterior(state, j, f_k, rng);

    util::validate_probability_vector(p_t);
    size_t word = common::util::sample_discrete(p_t, rng);
    size_t t_new = state.using_t[j][word];
    if (t_new == 0)
    {
        std::vector<float> p_k = calc_dish_posterior_w(state, f_k, rng);
        util::validate_probability_vector(p_k);
        size_t topic_index = common::util::sample_discrete(p_k, rng);
        size_t k_new = state.dishes_[topic_index];
        if (k_new == 0)
        {
            k_new = state.create_dish();
        }
        t_new = state.create_table(j, k_new);
    }
    state.add_table(j, t_new, i);
}

void
sampling_k(microscopes::lda::state &state, size_t j, size_t t, common::rng_t &rng) {
    state.leave_from_dish(j, t);
    std::vector<float> p_k = calc_dish_posterior_t(state, j, t, rng);
    util::validate_probability_vector(p_k);
    assert(state.dishes_.size() == p_k.size());
    size_t topic_index = common::util::sample_discrete(p_k, rng);
    size_t k_new = state.dishes_[topic_index];
    if (k_new == 0)
    {
        k_new = state.create_dish();
    }
    state.seat_at_dish(j, t, k_new);
}
} // namespace lda_crp

void
lda_crp_gibbs(microscopes::lda::state &state, common::rng_t &rng)
{
    for (size_t j = 0; j < state.x_ji.size(); ++j) {
        for (size_t i = 0; i < state.x_ji[j].size(); ++i) {
            lda_crp::sampling_t(state, j, i, rng);
        }
    }
    for (size_t j = 0; j < state.x_ji.size(); ++j) {
        for (auto t : state.using_t[j]) {
            if (t != 0) {
                lda_crp::sampling_k(state, j, t, rng);
            }
        }
    }
}

} // namespace kernels
} // namespace microscopes