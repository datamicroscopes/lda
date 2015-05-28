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
        rng_ = rng;
        for(size_t i = 0; i < M; ++i) {
            using_t.push_back({{0, true}});
        }
        using_k = {{0, true}};

        x_ji = std::vector<std::vector<size_t>>(docs);
        for(size_t j = 0; j < M; ++j) {
            k_jt.push_back({0});
            n_jt.push_back({0});

            n_jtv.push_back(std::vector< std::map<size_t, size_t>>());
            for (size_t t = 0; t < using_t.size(); ++t)
            {
                std::map<size_t, size_t> term_dict;
                for (size_t i = 0; i < V; ++i)
                {
                    term_dict[i] = 0;
                }
                n_jtv[j].push_back(term_dict);
            }
        }

        m = 0;
        m_k = std::vector<size_t> {1};
        n_k = std::vector<float> {beta_ * V};

        std::map<size_t, size_t> term_count;
        for (size_t i = 0; i < V; ++i)
        {
            term_count[i] = 0;
        }
        n_kv.push_back(term_count);

        for(size_t i=0; i < docs.size(); i++){

            t_ji.push_back(std::vector<size_t>(docs[i].size(), 0));
        }

    }

    void
    inference(){
        // for j, x_i in enumerate(self.x_ji):
        //     for i in xrange(len(x_i)):
        //         self.sampling_t(j, i)
        // for j in xrange(self.M):
        //     for t in self.using_t[j]:
        //         if t != 0: self.sampling_k(j, t)
    }

    void
    sampling_t(size_t j, size_t i){
        leave_from_table(j, i);
        size_t v = x_ji[j][i];
        std::vector<float> f_k = calc_f_k(v);

        // assert f_k[0] == 0 # f_k[0] is a dummy and will be erased
        std::vector<float> p_t = calc_table_posterior(j, f_k);
        // if len(p_t) > 1 and p_t[1] < 0: self.dump()
        size_t word = common::util::sample_discrete(p_t, rng_);
        size_t t_new = using_t[j][word];
        if (t_new == 0)
        {
            // float p_k = calc_dish_posterior_w(f_k);
            // k_new = self.using_k[numpy.random.multinomial(1, p_k).argmax()]
            // if k_new == 0:
            //     k_new = self.add_new_dish()
            // t_new = self.add_new_table(j, k_new)
        }

    }


private:
    // std::vector<float>
    // calc_dish_posterior_w(std::vector<float> f_k){
    //     std::map<size_t, float> p_k_map;
    //     for(auto k: using_k){
    //         p_k_map[k] = m_k[k] + f_k[k];
    //     }
    //     float sum_p_k = 0;
    //     for(auto& kv: p_k_map){
    //         p_k_map += kv.second;
    //     }
    //     std::vector<float> p_k;
    //     for (size_t i = 0; i < p_k_map.size(); ++i)
    //     {
    //         p_k.push_back(p_k_map[i] / sum_p_k);
    //     }
    //     return p_k;
    // }

    std::vector<float>
    calc_table_posterior(size_t j, std::vector<float> f_k){
        std::map<size_t, bool> using_table = using_t[j];
        std::map<size_t, float> p_t_map;
        for(auto& kv: using_table){
            p_t_map[kv.first] = n_jt[j][kv.first] + f_k[k_jt[j][kv.first]];
        }
        float p_x_ji = gamma_ / (float)V;
        for (size_t k = 0; k < f_k.size(); ++k)
        {
            p_x_ji += m_k[k] * f_k[k];
        }
        float sum_p_t = 0;
        for(auto& kv: p_t_map){
            sum_p_t += kv.second;
        }
        std::vector<float> p_t;
        for (size_t i = 0; i < p_t_map.size(); ++i)
        {
            p_t.push_back(p_t_map[i] / sum_p_t);
        }
        return p_t;
    }


    void
    leave_from_table(size_t j, size_t i){
        size_t t = t_ji[j][i];
        if (t > 0)
        {
            size_t k = k_jt[j][t];
            // decrease counters
            size_t v = x_ji[j][i];
            n_kv[k][v] -= 1;
            n_k[k] -= 1;
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
        using_t[j].erase(t);
        m_k[k] -= 1;
        m -= 1;
        if (m_k[k] == 0)
        {
            using_k.erase(k);
        }
    }

    std::vector<float>
    calc_f_k(size_t v){
        std::vector<float> f_k {};

        for (size_t k=0; k < n_kv.size(); k++)
        {
            f_k.push_back(n_kv[k][v] / n_k[k]);
        }

        return f_k;
    }

    size_t V; // Vocabulary size
    size_t M; // Num documents
    size_t m;
    float alpha_;
    float beta_;
    float gamma_;
    common::rng_t rng_;
    std::vector<std::map<size_t, bool>> using_t;
    std::map<size_t, bool> using_k;
    std::vector<std::vector<size_t>> x_ji;
    std::vector<std::vector<size_t>> k_jt;
    std::vector<std::vector<size_t>> n_jt;
    std::vector<std::vector<std::map<size_t, size_t>>> n_jtv;
    std::vector<size_t> m_k;
    std::vector<float> n_k;
    std::vector<std::map<size_t, size_t>> n_kv;
    std::vector<std::vector<size_t>> t_ji;


};

}
}