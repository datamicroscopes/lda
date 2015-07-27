#pragma once

#include <microscopes/common/util.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/lda/util.hpp>

#include <math.h>
#include <vector>
#include <set>
#include <map>

namespace microscopes {
namespace lda {


class model_definition {
public:
    model_definition(size_t, size_t);
    inline size_t n() const { return n_; }
    inline size_t v() const { return v_; }
private:
    size_t n_;
    size_t v_;
};

class state {
public:
    size_t V; // Size of vocabulary
    float alpha_; //  Hyperparamter on second level Dirichlet process
    float beta_; // Hyperparameter on base Dirichlet process
    float gamma_; // Hyperparameter on first level Dirichlet process
    std::vector<std::vector<size_t>> using_t; // table index (t=0 means to draw a new table)
    std::vector<size_t> dishes_; // using_k; dish(topic) index (k=0 means to draw a new dish)
    const std::vector<std::vector<size_t>> x_ji; // vocabulary for each document and term
    std::vector<std::vector<size_t>> restaurants_; // topics of document and table
    std::vector<std::vector<size_t>> n_jt; // number of terms for each table of document
    std::vector<std::vector<std::map<size_t, size_t>>> n_jtv; // number of occurrences of each term for each table of document
    std::vector<size_t> m_k; // number of tables for each topic
    lda_util::defaultdict<size_t, float> n_k; // number of terms for each topic ( + beta * V )
    std::vector<lda_util::defaultdict<size_t, float>> n_kv; // number of terms for each topic and vocabulary ( + beta )
    std::vector<std::vector<size_t>> table_doc_word; // t_ji table for each document and term (-1 means not-assigned)

    template <class... Args>
    static inline std::shared_ptr<state>
    initialize(Args &&... args)
    {
        return std::make_shared<state>(std::forward<Args>(args)...);
    }

private:
    state(const model_definition &defn,
          float alpha,
          float beta,
          float gamma,
          const std::vector<std::vector<size_t>> &docs,
          common::rng_t &);

public:
    state(const model_definition &defn,
          float alpha,
          float beta,
          float gamma,
          size_t initial_dishes,
          const std::vector<std::vector<size_t>> &docs,
          common::rng_t &);

    state(const model_definition &defn,
          float alpha,
          float beta,
          float gamma,
          const std::vector<std::vector<size_t>> &dish_assignments,
          const std::vector<std::vector<size_t>> &table_assignments,
          const std::vector<std::vector<size_t>> &docs,
          common::rng_t &);

    std::vector<std::vector<size_t>>
    assignments();

    /**
    * Returns, for each entity, a map from
    * table IDs -> (global) dish assignments
    *
    */
    std::vector<std::vector<size_t>>
    dish_assignments();

    /**
    * Returns, for each entity, an assignment vector
    * from each word to the (local) table it is assigned to.
    *
    */
    std::vector<std::vector<size_t>>
    table_assignments();

    // Not implemented
    float
    score_assignment() const;

    // Not implemented
    float
    score_data(common::rng_t &rng) const;

    std::vector<std::map<size_t, float>>
    word_distribution();

    std::vector<std::vector<float>>
    document_distribution();

    double
    perplexity();

    void
    leave_from_dish(size_t j, size_t t);

    void
    validate_n_k_values();

    void
    seat_at_dish(size_t j, size_t t, size_t k_new);

    void
    add_table(size_t ein, size_t t_new, size_t did);

    void
    create_entity();

    size_t
    create_dish();

    size_t
    create_table(size_t ein, size_t k_new);

    void
    remove_table(size_t eid, size_t tid);

    void
    delete_table(size_t eid, size_t tid);

    inline size_t tablesize(size_t eid, size_t tid) const { return n_jt[eid][tid]; }

    inline void delete_dish(size_t did) { lda_util::removeFirst(dishes_, did); }

    inline std::vector<size_t> dishes() const { return dishes_; }

    inline std::vector<size_t> tables(size_t eid) { return using_t[eid]; }

    inline size_t nentities() const { return x_ji.size(); }

    inline size_t ntopics() const { return dishes_.size() - 1; }

    inline size_t nwords() const { return V; }

    inline size_t nterms(size_t eid) const { return x_ji[eid].size(); }

    inline size_t ntables(size_t eid) const { return using_t[eid].size(); }

    inline std::vector<size_t> tables(size_t eid) const { return using_t[eid]; }

    inline int ntables() const { return std::accumulate(m_k.begin()+1, m_k.end(), 0); }

};

}
}