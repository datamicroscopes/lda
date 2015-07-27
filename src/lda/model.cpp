#include <microscopes/lda/model.hpp>


microscopes::lda::model_definition::model_definition(size_t n, size_t v)
    : n_(n), v_(v)
{
    MICROSCOPES_DCHECK(n > 0, "no docs");
    MICROSCOPES_DCHECK(v > 0, "no terms");
}

microscopes::lda::state::state(const model_definition &defn,
      float alpha,
      float beta,
      float gamma,
      const std::vector<std::vector<size_t>> &docs,
      common::rng_t &rng)
    : V(defn.v()),
      alpha_(alpha),
      beta_(beta),
      gamma_(gamma),
      x_ji(docs),
      n_k(lda_util::defaultdict<size_t, float>(beta * defn.v()))
      {
        // This page intentionally left blank
}

microscopes::lda::state::state(const model_definition &defn,
      float alpha,
      float beta,
      float gamma,
      size_t initial_dishes,
      const std::vector<std::vector<size_t>> &docs,
      common::rng_t &rng)
    : state(defn, alpha, beta, gamma, docs, rng) {

    auto dish_pool = microscopes::common::util::range(initial_dishes);

    create_dish(); // Dummy dish
    for (size_t eid = 0; eid < nentities(); ++eid) {
        create_entity();

        auto did = common::util::sample_choice(dish_pool, rng);
        if (did > dishes_.back()){
            did = create_dish();
        }
        create_table(eid, did);
        table_doc_word.push_back(std::vector<size_t>(nterms(eid), 0));
    }
}

microscopes::lda::state::state(const model_definition &defn,
      float alpha,
      float beta,
      float gamma,
      const std::vector<std::vector<size_t>> &dish_assignments,
      const std::vector<std::vector<size_t>> &table_assignments,
      const std::vector<std::vector<size_t>> &docs,
      common::rng_t &rng)
    : state(defn, alpha, beta, gamma, docs, rng) {

}

void
microscopes::lda::state::create_entity(){
    using_t.push_back(std::vector<size_t>());
    n_jt.push_back(std::vector<size_t>());
    restaurants_.push_back(std::vector<size_t>());
    n_jtv.push_back(std::vector< std::map<size_t, size_t>>());
}

std::vector<std::vector<size_t>>
microscopes::lda::state::assignments() {
    std::vector<std::vector<size_t>> ret;
    ret.resize(nentities());

    for (size_t eid = 0; eid < nentities(); eid++) {
        ret[eid].resize(table_doc_word[eid].size());
        for (size_t did = 0; did < table_doc_word[eid].size(); did++) {
            auto table = table_doc_word[eid][did];
            ret[eid][did] = restaurants_[eid][table];
        }
    }
    return ret;

}

/**
* Returns, for each entity, a map from
* table IDs -> (global) dish assignments
*
*/
std::vector<std::vector<size_t>>
microscopes::lda::state::dish_assignments() {
    return restaurants_;
}

/**
* Returns, for each entity, an assignment vector
* from each word to the (local) table it is assigned to.
*
*/
std::vector<std::vector<size_t>>
microscopes::lda::state::table_assignments() {
    return table_doc_word;
}

float
microscopes::lda::state::score_assignment() const
{
    return 0;
}

float
microscopes::lda::state::score_data(common::rng_t &rng) const
{
    return 0;
}


std::vector<std::map<size_t, float>>
microscopes::lda::state::word_distribution() {
    // Distribution over words for each topic
    std::vector<std::map<size_t, float>> vec;
    vec.reserve(dishes_.size());
    for (auto k : dishes_) {
        if (k == 0) continue;
        vec.push_back(std::map<size_t, float>());
        for (size_t v = 0; v < V; ++v) {
            if (n_kv[k].contains(v)) {
                vec.back()[v] = n_kv[k].get(v) / n_k.get(k);
            }
            else {
                vec.back()[v] = beta_ / n_k.get(k);
            }
        }
    }
    return vec;
}

std::vector<std::vector<float>>
microscopes::lda::state::document_distribution  () {
    // Distribution over topics for each document
    std::vector<std::vector<float>> theta;
    theta.reserve(restaurants_.size());
    std::vector<float> am_k(m_k.begin(), m_k.end());
    am_k[0] = gamma_;
    double sum_am_dishes_ = 0;
    for (auto k : dishes_) {
        sum_am_dishes_ += am_k[k];
    }
    for (size_t i = 0; i < am_k.size(); ++i) {
        am_k[i] *= alpha_ / sum_am_dishes_;
    }

    for (size_t j = 0; j < restaurants_.size(); j++) {
        std::vector<size_t> &n_jt_ = n_jt[j];
        std::vector<float> p_jk = am_k;
        for (auto t : using_t[j]) {
            if (t == 0) continue;
            size_t k = restaurants_[j][t];
            p_jk[k] += n_jt_[t];
        }
        p_jk = lda_util::selectByIndex(p_jk, dishes_);
        lda_util::normalize<float>(p_jk);
        theta.push_back(p_jk);
    }
    return theta;
}

double
microscopes::lda::state::perplexity() {
    std::vector<std::map<size_t, float>> phi = word_distribution();
    std::vector<std::vector<float>> theta = document_distribution();
    phi.insert(phi.begin(), std::map<size_t, float>());
    double log_likelihood = 0;
    size_t N = 0;
    for (size_t j = 0; j < x_ji.size(); j++) {
        auto &py_x_ji = x_ji[j];
        auto &p_jk = theta[j];
        for (auto &v : py_x_ji) {
            double word_prob = 0;
            for (size_t i = 0; i < p_jk.size(); i++) {
                auto p = p_jk[i];
                auto &p_kv = phi[i];
                word_prob += p * p_kv[v];
            }
            log_likelihood -= distributions::fast_log(word_prob);
        }
        N += x_ji[j].size();
    }

    return exp(log_likelihood / N);
}


// private:

void
microscopes::lda::state::leave_from_dish(size_t j, size_t t) {
    size_t k = restaurants_[j][t];
    MICROSCOPES_DCHECK(k > 0, "k < = 0");
    MICROSCOPES_DCHECK(m_k[k] > 0, "m_k[k] <= 0");
    m_k[k] -= 1; // one less table for topic k
    if (m_k[k] == 0) // destroy table
    {
        delete_dish(k);
        restaurants_[j][t] = 0;
    }
}

void
microscopes::lda::state::validate_n_k_values() {
    return;
    std::map<size_t, std::tuple<float, float>> values;
    for (auto k : dishes_) {
        float n_kv_sum = 0;
        for (size_t v = 0; v < V; v++) {
            n_kv_sum += n_kv[k].get(v);
        }
        values[k] = std::tuple<float, float>(n_kv_sum, n_k.get(k));
    }
    for (auto kv : values) {
        if (kv.first == 0) continue;
        MICROSCOPES_CHECK(std::abs((std::get<0>(kv.second) - std::get<1>(kv.second))) < 0.01,
                          "n_kv doesn't match n_k");
    }
}


void
microscopes::lda::state::seat_at_dish(size_t j, size_t t, size_t k_new) {
    m_k[k_new] += 1;

    size_t k_old = restaurants_[j][t];
    if (k_new != k_old)
    {
        MICROSCOPES_DCHECK(k_new != 0, "k_new is 0");
        restaurants_[j][t] = k_new;
        float n_jt_val = n_jt[j][t];

        if (k_old != 0)
        {
            n_k.decr(k_old, n_jt_val);
        }
        n_k.incr(k_new, n_jt_val);
        for (auto kv : n_jtv[j][t]) {
            auto v = kv.first;
            auto n = kv.second;
            MICROSCOPES_DCHECK(v < nwords(), "Word out of bounds");
            if (k_old != 0)
            {
                n_kv[k_old].decr(v, n);
            }
            n_kv[k_new].incr(v, n);
        }
    }
}


void
microscopes::lda::state::add_table(size_t ein, size_t t_new, size_t did) {
    table_doc_word[ein][did] = t_new;
    n_jt[ein][t_new] += 1;

    size_t k_new = restaurants_[ein][t_new];
    n_k.incr(k_new, 1);

    size_t v = x_ji[ein][did];
    MICROSCOPES_DCHECK(v < nwords(), "Word out of bounds");
    n_kv[k_new].incr(v, 1);
    n_jtv[ein][t_new][v] += 1;
}

size_t
microscopes::lda::state::create_dish() {
    size_t k_new = dishes_.size();
    for (size_t i = 0; i < dishes_.size(); ++i)
    {
        if (i != dishes_[i])
        {
            k_new = i;
            break;
        }
    }
    if (k_new == dishes_.size())
    {
        m_k.push_back(0);
        n_kv.push_back(lda_util::defaultdict<size_t, float>(beta_));
        MICROSCOPES_DCHECK(dishes_.size() == 0 || k_new == dishes_.back() + 1, "bad k_new (1)");
        MICROSCOPES_DCHECK(k_new < n_kv.size(), "bad k_new (2)");
    }

    dishes_.insert(dishes_.begin() + k_new, k_new);
    n_k.set(k_new, beta_ * V);
    n_kv[k_new] = lda_util::defaultdict<size_t, float>(beta_);
    m_k[k_new] = 0;
    return k_new;

}

size_t
microscopes::lda::state::create_table(size_t ein, size_t k_new)
{
    size_t t_new = using_t[ein].size();
    for (size_t i = 0; i < using_t[ein].size(); ++i)
    {
        if (i != using_t[ein][i])
        {
            t_new = i;
            break;
        }
    }
    if (t_new == using_t[ein].size())
    {
        n_jt[ein].push_back(0);
        restaurants_[ein].push_back(0);

        n_jtv[ein].push_back(std::map<size_t, size_t>());
    }
    using_t[ein].insert(using_t[ein].begin() + t_new, t_new);
    n_jt[ein][t_new] = 0;
    // MICROSCOPES_DCHECK(k_new != 0, "k_new ");
    restaurants_[ein][t_new] = k_new;
    if (k_new != 0){
        m_k[k_new] += 1;
    }

    return t_new;
}

void
microscopes::lda::state::remove_table(size_t eid, size_t tid) {
    size_t t = table_doc_word[eid][tid];
    if (t > 0)
    {
        size_t k = restaurants_[eid][t];
        MICROSCOPES_DCHECK(k > 0, "k <= 0");
        // decrease counters
        size_t v = x_ji[eid][tid];
        MICROSCOPES_DCHECK(v < nwords(), "Word out of bounds");
        n_kv[k].decr(v, 1);
        n_k.decr(k, 1);
        n_jt[eid][t] -= 1;
        n_jtv[eid][t][v] -= 1;

        if (n_jt[eid][t] == 0)
        {
            delete_table(eid, t);
        }
    }
}

void
microscopes::lda::state::delete_table(size_t eid, size_t tid) {
    size_t k = restaurants_[eid][tid];
    lda_util::removeFirst(using_t[eid], tid);
    m_k[k] -= 1;
    MICROSCOPES_DCHECK(m_k[k] >= 0, "m_k[k] < 0");
    if (m_k[k] == 0)
    {
        delete_dish(k);
    }
}
