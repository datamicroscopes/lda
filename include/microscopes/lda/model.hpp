#pragma once

#include <microscopes/models/base.hpp>
#include <microscopes/common/entity_state.hpp>
#include <microscopes/common/group_manager.hpp>
#include <microscopes/common/recarray/dataview.hpp>
#include <microscopes/common/util.hpp>
#include <microscopes/common/typedefs.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/io/schema.pb.h>
#include <distributions/special.hpp>

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
          const common::hyperparam_bag_t &topic_init,
          const common::hyperparam_bag_t &word_init,
          const common::recarray::dataview &data, // should be variadic
          const std::vector<std::vector<size_t>> &dish_assignments,
          const std::vector<std::vector<size_t>> &table_assigments,
          common::rng_t &rng)
    {
    }

    // static std::shared_ptr<state>
    // unsafe_initialize(const model_definition &def)
    // {
    //     return std::make_shared<state>();
    // }
protected:
    std::vector<std::shared_ptr<models::hypers>> hypers_;
    common::group_manager<group_type> groups_;
};

}
}