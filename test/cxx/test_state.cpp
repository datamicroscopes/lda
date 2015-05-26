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

int main(void){
    const size_t D = 28*28;
    rng_t r(5849343);

    vector<shared_ptr<models::model>> models;
    for (size_t i = 0; i < D; i++){
        models.emplace_back(make_shared<
            models::distributions_model<BetaBernoulli>>());
    }

    lda::model_definition def(50, 100);

    return 0;
}