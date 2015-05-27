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
    static const size_t MaxVocabularySize = 0x10000;
    typedef distributions::DirichletDiscrete<MaxVocabularySize> DD;

    typedef io::CRP dish_message_type;
    typedef distributions::protobuf::DirichletDiscrete_Shared vocab_message_type;

    state(const model_definition &def,
          const std::string &topic_init, // std::string == common::hyperparam_bag_t
          const std::string &word_init,  // std::string == common::hyperparam_bag_t
          const common::variadic::dataview &data,
          const std::vector<std::vector<size_t>> &dish_assignments,
          const std::vector<std::vector<size_t>> &table_assigments,
          common::rng_t &rng)
    {
        // Didn't finish yanking from https://github.com/datamicroscopes/lda/blob/dcfa0a5462ea34abf2d39ffb6692ca7b3e8371ae/include/microscopes/lda/model.hpp#L109-L164
        // common_init(def, topic_init, word_init, data);

        // size_t num_dishes = 0;
        // for (const auto &p : dish_assignments){
        //     for (const auto &q : p){
        //         num_dishes = std::max(num_dishes, q);
        //     }
        // }
        // num_dishes += 1;

        // for(size_t i = 0; i < num_dishes; ++i) {
        //     auto p = dishes_.create_group();
        //     p.second.group_.init(shared_, rng);
        // }

    }

    state(const model_definition &def,
          const common::hyperparam_bag_t &topic_init,
          const common::hyperparam_bag_t &word_init,
          const common::variadic::dataview &data,
          size_t k,
          std::vector<std::vector<size_t>> &assignments,
          common::rng_t &rng) {
        // from https://github.com/datamicroscopes/lda/blob/dcfa0a5462ea34abf2d39ffb6692ca7b3e8371ae/include/microscopes/lda/model.hpp#L171-L196
        common_init(def, topic_init, word_init, data);

        std::vector<std::vector<size_t>> actual_assignments(assignments);

        if (actual_assignments.empty()) {
            actual_assignments.resize(def.n());
            // XXX(stephentu):
            // arbitrarily start with 10 tables per document
            std::uniform_int_distribution<unsigned> topic_dist(0, 9);
            for (unsigned i = 0; i < def.n(); ++i) {
                auto acc = data.get(i);
                actual_assignments[i].resize(acc.n());
                for (size_t j = 0; j < acc.n(); j++) {
                    const size_t k = topic_dist(rng);
                    actual_assignments[i][j] = k;
                }
            }
        }
        MICROSCOPES_DCHECK(actual_assignments.size() == def.n(),
                           "invalid size assignment vector");

        for (unsigned i = 0; i < k; ++i) {
            auto p = dishes_.create_group();
            p.second.group_.init(shared_, rng);
        }
        std::uniform_int_distribution<unsigned> dish_dist(0, k - 1);

        restaurants_.resize(def.n());
        for (size_t i = 0; i < def.n(); i++) {
            auto acc = data.get(i);
            MICROSCOPES_DCHECK(acc.n() > 0,
                               "empty documents are not allowed");
            restaurants_[i] = common::group_manager<restaurant_suffstat_t>(acc.n());
            MICROSCOPES_DCHECK(actual_assignments[i].size() == acc.n(),
                               "invalid size document assignment vector");
            const size_t ntables = *std::max_element(
                                       actual_assignments[i].begin(),
                                       actual_assignments[i].end()) + 1;
            for (size_t j = 0; j < ntables; j++) {
                auto p = restaurants_[i].create_group();
                MICROSCOPES_ASSERT(p.second.dish_ == -1);
                p.second.group_.init(shared_, rng);
            }
            MICROSCOPES_ASSERT(restaurants_[i].ngroups() == ntables);
            for (size_t j = 0; j < acc.n(); j++) {
                const size_t w = acc.get(j).get<uint32_t>();
                MICROSCOPES_DCHECK(w < def.v(), "invalid entry");
                auto &table_ref = restaurants_[i].add_value(actual_assignments[i][j], j);
                // XXX(stephentu): better dish assignment
                // randomly pick the dish
                if (table_ref.dish_ == -1)
                    table_ref.dish_ = dish_dist(rng);
                table_ref.group_.add_value(shared_, w, rng);
                auto &dish = dishes_.group(table_ref.dish_);
                dish.group_.add_value(shared_, w, rng);
            }
        }
    }

    // static std::shared_ptr<state>
    // unsafe_initialize(const model_definition &def)
    // {
    //     return std::make_shared<state>();
    // }

private:
    void
    common_init(const model_definition &def,
                const std::string &topic_init,
                const std::string &word_init,
                const common::variadic::dataview &data) {
        MICROSCOPES_DCHECK(def.v() <= MaxVocabularySize, "vocab too large");
        MICROSCOPES_DCHECK(data.size() == def.n(), "data mismatch");

        dish_message_type topic_m;
        common::util::protobuf_from_string(topic_m, topic_init);
        MICROSCOPES_DCHECK(topic_m.alpha() > 0., "invalid alpha");
        alpha_ = topic_m.alpha();

        vocab_message_type word_m;
        common::util::protobuf_from_string(topic_m, topic_init);
        MICROSCOPES_DCHECK((size_t)word_m.alphas_size() == def.v(),
                           "word mismatch");
        shared_.dim = def.v();
        for (size_t i = 0 ; i < def.v(); i++) {
            MICROSCOPES_DCHECK(word_m.alphas(i) > 0., "invalid alpha found");
            shared_.alphas[i] = word_m.alphas(i);
            shared_alphas_sum_ += word_m.alphas(i);
        }
    }

    struct dish_suffstat_t {
        DD::Group group_;
    };

    struct restaurant_suffstat_t {
        restaurant_suffstat_t() : group_(), dish_(-1) {}

        DD::Group group_;
        ssize_t dish_;
    };

    float alpha_; // hyperparam for the top level dDP
    distributions::DirichletDiscrete<MaxVocabularySize>::Shared shared_; // hyperparam on the base measure of the individual Dps
    float shared_alphas_sum_; // normalization constant to turn the
    // base measure dirichlet alphas into
    // a probability distribution
    common::simple_group_manager<dish_suffstat_t> dishes_;
    std::vector<common::group_manager<restaurant_suffstat_t>> restaurants_;
};

}
}