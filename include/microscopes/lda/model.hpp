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

#include <cmath>
#include <vector>
#include <set>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <utility>
#include <stdexcept>


namespace microscopes {
namespace lda {

typedef std::vector<std::shared_ptr<models::group>> group_type;



class model_definition {
public:
    model_definition(size_t n, size_t v)
        : n_(n), v_(v)
    {
        MICROSCOPES_DCHECK(n > 0, "no docs");
        MICROSCOPES_DCHECK(v > 0, "no terms");
    }

    std::vector<std::shared_ptr<models::hypers>>
            create_hypers() const
    {
        std::vector<common::runtime_type> ret = std::vector<common::runtime_type>();
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
            : alpha_(alpha), beta_(beta), gamma_(gamma) {
        V = def.v();
        M = def.n();

        for(size_t i = 0; i < M; ++i) {
            using_t.push_back({0});
        }
        using_k = std::vector<size_t> {0};

        x_ji = std::vector<std::vector<size_t>>(docs);
        for(size_t j = 0; j < M; ++j) {
            k_jt.push_back({0});
            n_jt.push_back({0});
            n_jtv.push_back({0});
        }

        m = 0;
        m_k = std::vector<size_t> {1};
        n_k = std::vector<float> {beta_ * V};
        // // n_kv =

        for(size_t i=0; i < docs.size(); i++){
            t_ji.push_back(std::vector<size_t>(docs[i].size(), 0));
        }



    }

private:
    size_t V; // Vocabulary size
    size_t M; // Num documents
    size_t m;
    float alpha_;
    float beta_;
    float gamma_;
    std::vector<std::vector<size_t>> using_t;
    std::vector<size_t> using_k;
    std::vector<std::vector<size_t>> x_ji;
    std::vector<std::vector<size_t>> k_jt;
    std::vector<std::vector<size_t>> n_jt;
    std::vector<std::vector<size_t>> n_jtv;
    std::vector<size_t> m_k;
    std::vector<float> n_k;
    // std::vector<std::map<key, value> map; > n_kv;
    std::vector<std::vector<size_t>> t_ji;


};

}
}