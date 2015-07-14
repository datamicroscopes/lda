#include <microscopes/lda/model.hpp>


microscopes::lda::model_definition::model_definition(size_t n, size_t v)
    : n_(n), v_(v)
{
    MICROSCOPES_DCHECK(n > 0, "no docs");
    MICROSCOPES_DCHECK(v > 0, "no terms");
}

microscopes::lda::state::state(const model_definition &def,
      float alpha,
      float beta,
      float gamma,
      const std::vector<std::vector<size_t>> &docs,
      common::rng_t &rng)
    : alpha_(alpha), beta_(beta), gamma_(gamma), x_ji(docs),
      n_k(util::defaultdict<size_t, float>(beta * def.v())) {
    V = def.v();
    for (size_t i = 0; i < x_ji.size(); ++i) {
        using_t.push_back({0});
    }
    dishes_ = {0};

    for (size_t j = 0; j < x_ji.size(); ++j) {
        restaurants_.push_back({0});
        n_jt.push_back({0});

        n_jtv.push_back(std::vector< std::map<size_t, size_t>>());
        for (size_t t = 0; t < using_t[j].size(); ++t)
        {
            n_jtv[j].push_back(std::map<size_t, size_t>());
        }
    }
    m = 0;
    m_k = std::vector<size_t> {1};
    n_kv.push_back(util::defaultdict<size_t, float>(beta_));
    for (size_t i = 0; i < docs.size(); i++) {

        t_ji.push_back(std::vector<size_t>(docs[i].size(), 0));
    }
}
