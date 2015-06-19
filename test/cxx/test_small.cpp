#include <microscopes/lda/model.hpp>
#include <microscopes/lda/data.hpp>
#include <microscopes/lda/random_docs.hpp>
#include <microscopes/common/macros.hpp>
#include <microscopes/common/recarray/dataview.hpp>
#include <microscopes/models/distributions.hpp>
#include <microscopes/common/random_fwd.hpp>

#include <random>
#include <iostream>
#include <time.h>

using namespace std;
using namespace distributions;
using namespace microscopes;
using namespace microscopes::common;
using namespace microscopes::common::recarray;


static void
test_small()
{
    rng_t r(time(NULL)); //

    std::cout << "creating model definition...";
    lda::model_definition def(3, 4);
    std::vector<std::vector<size_t>> docs = {{0, 1, 2}, {1, 2, 3}, {3, 3}};
    lda::state state(def, .1, .5, .1, docs, r);
    std::cout << " complete" << std::endl;

    for(unsigned i = 0; i < 100; ++i){
        std::cout << "inference step: " << i << std::endl;
        state.inference();
        std::cout << "   K=" << state.usedDishes() << std::endl;
        std::cout << "   p=" << state.perplexity() << std::endl;
        std::map<size_t, int> count;
        for(auto k_j: state.k_jt){
            for(auto k: k_j){
                count[k]++;
            }
        }
        std::cout << "k_jt count  " << count << std::endl;
    }
    std::cout << "FINI!" << std::endl;
}

int main(void){
    test_small();
}