#include <microscopes/lda/model.hpp>
#include <microscopes/common/recarray/dataview.hpp>
#include <microscopes/models/distributions.hpp>
#include <microscopes/common/random_fwd.hpp>

#include <random>
#include <iostream>

using namespace std;
using namespace distributions;
using namespace microscopes;
using namespace microscopes::common;
using namespace microscopes::common::recarray;


static void
test_create_model_def_and_state()
{
    const size_t D = 28*28;
    rng_t r(5849343);

    vector<shared_ptr<models::model>> models;
    for (size_t i = 0; i < D; i++){
        models.emplace_back(make_shared<
            models::distributions_model<BetaBernoulli>>());
    }

    lda::model_definition def(50, 100);
    std::vector< std::vector<size_t>> docs {{0, 1, 2}, {2, 3, 4, 1}};
    lda::state state(def, 1, .5, 1, docs, r);
    for(unsigned i = 0; i < 2000; ++i){
        state.inference();
        std::cout << "iter" << i << std::endl;
    }
    std::cout << "FINI!";
}


int main(void){
    test_create_model_def_and_state();
    return 0;
}