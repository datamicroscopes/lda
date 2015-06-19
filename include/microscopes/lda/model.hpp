#define _GLIBCXX_DEBUG
#include <microscopes/models/base.hpp>
#include <microscopes/common/entity_state.hpp>
#include <microscopes/common/group_manager.hpp>
#include <microscopes/common/variadic/dataview.hpp>
#include <microscopes/common/util.hpp>
#include <microscopes/common/typedefs.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/io/schema.pb.h>
#include <distributions/special.hpp>
#include <distributions/models/dd.hpp>
#include <eigen3/Eigen/dense>

#include <math.h>
#include <assert.h>
#include <vector>
#include <set>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <utility>
#include <stdexcept>

using namespace distributions;

namespace microscopes {
namespace lda {

typedef std::vector<std::shared_ptr<models::group>> group_type;

void
validate_probability_vector(const std::vector<float> &p){
    float sum = 0;
    for(auto x: p){
        assert(isfinite(x));
        assert(x >= 0);
        sum+=x;
    }
    assert(std::abs(1 - sum) < 0.01);
}

template<typename T> void
removeFirst(std::vector<T> &v, T element){
    auto it = std::find(v.begin(),v.end(), element);
    if (it != v.end()) {
      v.erase(it);
    }
}

// http://stackoverflow.com/a/1267878/982745
template< class T >
std::vector<T> selectByIndex(const std::vector<T> &v, const std::vector<size_t> &index )  {
    std::vector<T> new_v;
    new_v.reserve(index.size());
    for(size_t i: index){
        new_v.push_back(v[i]);
    }

    return new_v;
}

template<class T>
void
normalize(std::vector<T> &v){
    for(auto x: v) {
        assert(isfinite(x));
    }
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(v.data(), v.size());
    vec /= vec.sum();
}

class model_definition {
public:
    model_definition(size_t n, size_t v)
        : n_(n), v_(v)
    {
        MICROSCOPES_DCHECK(n > 0, "no docs");
        MICROSCOPES_DCHECK(v > 0, "no terms");
    }


    inline size_t n() const { return n_; }
    inline size_t v() const { return v_; }
private:
    size_t n_;
    size_t v_;
};

class state {
public:
    // Quick and dirty constructor for Shuyo implementation.
    state(const model_definition &def,
        float alpha,
        float beta,
        float gamma,
        const std::vector<std::vector<size_t>> &docs,
        common::rng_t &rng)
            : alpha_(alpha), beta_(beta), gamma_(gamma), x_ji(docs) {
        V = def.v();
        rng_ = rng;
        for(size_t i = 0; i < x_ji.size(); ++i) {
            using_t.push_back({0});
        }
        using_k = {0};

        for(size_t j = 0; j < x_ji.size(); ++j) {
            k_jt.push_back({0});
            n_jt.push_back({0});

            n_jtv.push_back(std::vector< std::map<size_t, size_t>>());
            for (size_t t = 0; t < using_t[j].size(); ++t)
            {
                n_jtv[j].push_back(std::map<size_t, size_t>());
            }
        }
        m = 0;
        m_k = std::vector<size_t> {1};
        n_k = std::vector<float> {0};

        n_kv.push_back(std::map<size_t, float>());

        for(size_t i=0; i < docs.size(); i++){

            t_ji.push_back(std::vector<size_t>(docs[i].size(), 0));
        }

    }

    void
    inference(){
        for (size_t j = 0; j < x_ji.size(); ++j){
            for (size_t i = 0; i < x_ji[j].size(); ++i){
                sampling_t(j, i);
            }
        }
        for (size_t j = 0; j < x_ji.size(); ++j){
            for (auto t: using_t[j]){
                if(t != 0) {
                    sampling_k(j, t);
                }
            }
        }
    }

    size_t
    usedDishes(){
        return using_k.size() - 1;
    }


    std::vector<std::map<size_t, float>>
    wordDist(){
        // Distribution over words for each topic
        std::vector<std::map<size_t, float>> vec;
        vec.reserve(using_k.size());
        for(auto k: using_k){
            if(k==0) continue;
            vec.push_back(std::map<size_t, float>());
            for(size_t v = 0; v < V; ++v) {
                if(n_kv[k].find(v) != n_kv[k].end()){
                    vec.back()[v] = get_n_kv(k, v) / get_n_k(k);
                }
                else{
                    vec.back()[v] = beta_ / get_n_k(k);
                }
            }
        }
        return vec;
    }

    std::vector<std::vector<float>>
    docDist(){
        // Distribution over topics for each document
        std::vector<std::vector<float>> theta;
        theta.reserve(k_jt.size());
        std::vector<float> am_k(m_k.begin(), m_k.end());
        am_k[0] = gamma_;
        double sum_am_using_k = 0;
        for(auto k: using_k){
            sum_am_using_k += am_k[k];
        }
        for(size_t i = 0; i < am_k.size(); ++i) {
            am_k[i] *= alpha_ / sum_am_using_k;
        }

        for(size_t j = 0; j < k_jt.size(); j++){
            std::vector<size_t> &n_jt_ = n_jt[j];
            std::vector<float> p_jk = am_k;
            for(auto t: using_t[j]){
                if(t == 0) continue;
                size_t k = k_jt[j][t];
                p_jk[k] += n_jt_[t];
            }
            p_jk = selectByIndex(p_jk, using_k);
            normalize(p_jk);
            theta.push_back(p_jk);
        }
        return theta;
    }

    double
    perplexity(){
        std::vector<std::map<size_t, float>> phi = wordDist();
        std::vector<std::vector<float>> theta = docDist();
        phi.insert(phi.begin(), std::map<size_t, float>());
        double log_likelihood = 0;
        size_t N = 0;
        for(size_t j = 0; j < x_ji.size(); j++){
            auto &py_x_ji = x_ji[j];
            auto &p_jk = theta[j];
            for(auto &v: py_x_ji){
                double word_prob = 0;
                for(size_t i = 0; i < p_jk.size(); i++){
                    auto p = p_jk[i];
                    auto &p_kv = phi[i];
                    word_prob += p * p_kv[v];
                }
                log_likelihood -= fast_log(word_prob);
            }
            N += x_ji[j].size();
        }

        return exp(log_likelihood / N);
    }


// private:
    void
    sampling_t(size_t j, size_t i){
        leave_from_table(j, i);
        size_t v = x_ji[j][i];
        std::vector<float> f_k = calc_f_k(v);
        assert(f_k[0] == 0);
        std::vector<float> p_t = calc_table_posterior(j, f_k);
        // if len(p_t) > 1 and p_t[1] < 0: self.dump()
        validate_probability_vector(p_t);
        size_t word = common::util::sample_discrete(p_t, rng_);
        size_t t_new = using_t[j][word];
        if (t_new == 0)
        {
            std::vector<float> p_k = calc_dish_posterior_w(f_k);
            validate_probability_vector(p_k);
            size_t topic_index = common::util::sample_discrete(p_k, rng_);
            size_t k_new = using_k[topic_index];
            if (k_new == 0)
            {
                k_new = add_new_dish();
            }
            t_new = add_new_table(j, k_new);
        }
        seat_at_table(j, i, t_new);
    }

    void
    sampling_k(size_t j, size_t t){
        leave_from_dish(j, t);
        std::vector<float> p_k = calc_dish_posterior_t(j, t);
        validate_probability_vector(p_k);
        assert(using_k.size() == p_k.size());
        size_t topic_index = common::util::sample_discrete(p_k, rng_);
        size_t k_new = using_k[topic_index];
        if (k_new == 0)
        {
            k_new = add_new_dish();
        }
        seat_at_dish(j, t, k_new);
    }

    void
    leave_from_dish(size_t j, size_t t){
        size_t k = k_jt[j][t];
        assert(k > 0);
        assert(m_k[k] > 0);
        m_k[k] -= 1;
        m -= 1;
        if (m_k[k] == 0)
        {
            removeFirst(using_k, k);
            k_jt[j][t] = 0;
        }
    }

    void
    validate_n_k_values(){
        std::map<size_t, std::tuple<float, float>> values;
        for(auto k: using_k){
            float n_kv_sum = 0;
            for(size_t v = 0; v < V; v++){
                n_kv_sum += get_n_kv(k, v);
            }
            values[k] = {n_kv_sum, get_n_k(k)};
        }
        std::cout << "values :" << std::endl;
        for(auto v: values){
            std::cout << "    :" << std::get<0>(v.second) << "  " << std::get<1>(v.second) << std::endl;
        }
        for(auto kv: values){
            if (kv.first == 0) continue;
            assert(std::abs((std::get<0>(kv.second) - std::get<1>(kv.second))) < 0.01);
        }
    }

    std::vector<float>
    calc_dish_posterior_t(size_t j, size_t t){
        std::vector<float> log_p_k(using_k.size());
        for(size_t i = 0; i < using_k.size(); i++){
            auto n_k_val = (using_k[i] == k_jt[j][t]) ? get_n_k(i) - n_jt[j][t] : get_n_k(i);
            log_p_k[i] = fast_log(m_k[using_k[i]]) + fast_lgamma(n_k_val) - fast_lgamma(n_k_val + n_jt[j][t]);
            assert(!isinf(log_p_k[i]));
        }
        float log_p_k_new = fast_log(gamma_) + fast_lgamma(V * beta_) - fast_lgamma(V * beta_ + n_jt[j][t]);

        for(auto &kv: n_jtv[j][t]){
            auto w = kv.first;
            auto n_jtw = kv.second;
            if (n_jtw == 0) continue;
            assert(n_jtw > 0);

            std::vector<float> n_kw(using_k.size());
            for(size_t i = 0; i < using_k.size(); i++){
                n_kw[i] = get_n_kv(using_k[i], w);
                if(using_k[i] == k_jt[j][t]) n_kw[i] -= n_jtw;
                assert(i == 0 || n_kw[i] > 0);
            }
            n_kw[0] = 1; // # dummy for logarithm's warning
            for(size_t i = 0; i < n_kw.size(); i++){
                log_p_k[i] += fast_lgamma(n_kw[i] + n_jtw) - fast_lgamma(n_kw[i]);
            }
            log_p_k_new += fast_lgamma(beta_ + n_jtw) - fast_lgamma(beta_);
        }
        log_p_k[0] = log_p_k_new;
        for(auto x: log_p_k) assert(isfinite(x));

        std::vector<float> p_k;
        p_k.reserve(using_k.size());
        float max_value = *std::max_element(log_p_k.begin(), log_p_k.end());
        for(auto log_p_k_value: log_p_k){
            p_k.push_back(exp(log_p_k_value - max_value));
        }
        normalize(p_k);
        return p_k;
    }


    void
    seat_at_dish(size_t j, size_t t, size_t k_new){
        m += 1;
        m_k[k_new] += 1;

        size_t k_old = k_jt[j][t];
        if (k_new != k_old)
        {
            assert(k_new != 0);
            k_jt[j][t] = k_new;
            float n_jt_val = n_jt[j][t];

            if (k_old != 0)
            {
                decrement_n_k(k_old, n_jt_val);
            }
            increment_n_k(k_new, n_jt_val);
            for(auto kv: n_jtv[j][t]){
                auto v = kv.first;
                auto n = kv.second;
                if (k_old != 0)
                {
                    decrement_n_kv(k_old, v, n);
                }
                increment_n_kv(k_new, v, n);
            }
        }
    }


    void
    seat_at_table(size_t j, size_t i, size_t t_new){
        t_ji[j][i] = t_new;
        n_jt[j][t_new] += 1;

        size_t k_new = k_jt[j][t_new];
        increment_n_k(k_new, 1);

        size_t v = x_ji[j][i];
        increment_n_kv(k_new, v, 1);
        n_jtv[j][t_new][v] += 1;
    }


    size_t
    add_new_dish(){
        size_t k_new = using_k.size();
        for (size_t i = 0; i < using_k.size(); ++i)
        {
            if (i != using_k[i])
            {
                k_new = i;
                break;
            }
        }
        if (k_new == using_k.size())
        {
            n_k.push_back(0);
            m_k.push_back(m_k[0]);
            n_kv.push_back(std::map<size_t, float>());
            assert(k_new == using_k.back() + 1);
            assert(k_new < n_kv.size());
        }

        using_k.insert(using_k.begin()+k_new, k_new);
        n_k[k_new] = 0;
        n_kv[k_new] = std::map<size_t, float>();
        m_k[k_new] = 0;
        return k_new;

    }

    size_t
    add_new_table(size_t j, size_t k_new)
    {
        size_t t_new = using_t[j].size();
        for (size_t i = 0; i < using_t[j].size(); ++i)
        {
            if (i != using_t[j][i])
            {
                t_new = i;
                break;
            }
        }
        if (t_new == using_t[j].size())
        {
            n_jt[j].push_back(0);
            k_jt[j].push_back(0);

            n_jtv[j].push_back(std::map<size_t, size_t>());
        }
        using_t[j].insert(using_t[j].begin()+t_new, t_new);
        n_jt[j][t_new] = 0;
        assert(k_new != 0);
        k_jt[j][t_new] = k_new;
        m_k[k_new] += 1;
        m += 1;

        return t_new;
    }

    std::vector<float>
    calc_dish_posterior_w(const std::vector<float> &f_k){
        Eigen::VectorXf p_k(using_k.size());
        for(size_t i = 0; i < using_k.size(); ++i) {
            p_k(i) = m_k[using_k[i]] * f_k[using_k[i]];
        }
        p_k(0) = gamma_ / V;
        p_k /= p_k.sum();
        return std::vector<float>(p_k.data(), p_k.data() + p_k.size());
    }

    std::vector<float>
    calc_table_posterior(size_t j, std::vector<float> &f_k){
        std::vector<size_t> using_table = using_t[j];
        Eigen::VectorXf p_t(using_table.size());

        for(size_t i = 0; i < using_table.size(); i++){
            auto p = using_table[i];
            p_t(i) = n_jt[j][p] * f_k[k_jt[j][p]];
        }
        Eigen::Map<Eigen::VectorXf> eigen_f_k(f_k.data(), f_k.size());
        Eigen::Map<Eigen::Matrix<size_t, Eigen::Dynamic, 1>> eigen_m_k(m_k.data(), m_k.size());
        float p_x_ji = gamma_ / V + eigen_f_k.dot(eigen_m_k.cast<float>());
        p_t[0] = p_x_ji * alpha_ / (gamma_ + m);
        p_t /= p_t.sum();
        return std::vector<float>(p_t.data(), p_t.data() + p_t.size());
    }


    void
    leave_from_table(size_t j, size_t i){
        size_t t = t_ji[j][i];
        if (t > 0)
        {
            size_t k = k_jt[j][t];
            assert(k > 0);
            // decrease counters
            size_t v = x_ji[j][i];
            decrement_n_kv(k, v, 1);
            decrement_n_k(k, 1);
            n_jt[j][t] -= 1;
            n_jtv[j][t][v] -= 1;

            if (n_jt[j][t] == 0)
            {
                remove_table(j, t);
            }
        }
    }

    void
    remove_table(size_t j, size_t t){
        size_t k = k_jt[j][t];
        removeFirst(using_t[j], t);
        m_k[k] -= 1;
        m -= 1;
        assert(m_k[k] >= 0);
        if (m_k[k] == 0)
        {
            removeFirst(using_k, k);
        }
    }

    std::vector<float>
    calc_f_k(size_t v){
        Eigen::VectorXf f_k(n_kv.size());

        for (size_t k=0; k < n_kv.size(); k++)
        {
            f_k(k) = get_n_kv(k, v) / get_n_k(k);
        }

        return std::vector<float>(f_k.data(), f_k.data() + f_k.size());
    }

    float
    get_n_k(size_t k){
        if (n_k[k] > 0){
          return n_k[k];
        }
        else {
            return V * beta_;
        }
    }

    void
    increment_n_k(size_t k, float amount){
        n_k[k] += amount;
        if (n_k[k] == amount){
            n_k[k] += V * beta_;
        }
    }

    void
    decrement_n_k(size_t k, float amount){
        n_k[k] -= amount;
        if (n_k[k] == -amount){
            n_k[k] += V * beta_;
        }
    }

    float
    get_n_kv(size_t k, size_t v){
        if ((n_kv[k].count(v) > 0 && n_kv[k][v] > 0) || k == 0){
          return n_kv[k][v];
        }
        else {
            return beta_;
        }
    }

    void
    increment_n_kv(size_t k, size_t v, float amount){
        n_kv[k][v] += amount;
        if (n_kv[k][v] == amount){
            n_kv[k][v] += beta_;
        }
    }

    void
    decrement_n_kv(size_t k, size_t v, float amount){
        n_kv[k][v] -= amount;
        if (n_kv[k][v] == -amount){
            n_kv[k][v] += beta_;
        }
    }

    size_t V; // Vocabulary size
    // size_t x_ji.size(); // Num documents
    size_t m;
    float alpha_;
    float beta_;
    float gamma_;
    common::rng_t rng_;
    std::vector<std::vector<size_t>> using_t;
    std::vector<size_t> using_k;
    const std::vector<std::vector<size_t>> &x_ji;
    std::vector<std::vector<size_t>> k_jt;
    std::vector<std::vector<size_t>> n_jt;
    std::vector<std::vector<std::map<size_t, size_t>>> n_jtv;
    std::vector<size_t> m_k;
    std::vector<float> n_k;
    std::vector<std::map<size_t, float>> n_kv;
    std::vector<std::vector<size_t>> t_ji;


};

}
}