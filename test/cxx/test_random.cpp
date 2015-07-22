#include <microscopes/lda/model.hpp>
#include <microscopes/lda/kernels.hpp>
#include <microscopes/lda/random_docs.hpp>
#include <microscopes/common/macros.hpp>
#include <microscopes/models/distributions.hpp>
#include <microscopes/common/random_fwd.hpp>

#include <random>
#include <iostream>

using namespace std;
using namespace distributions;
using namespace microscopes;
using namespace microscopes::common;


static void
sequence_random(double alpha, double beta, double gamma, size_t seed){
    std::cout << alpha << " " << beta << " " << gamma <<std::endl;
    rng_t r(seed);
    std::vector< std::vector<size_t>> docs {{0,1,2,3}, {0,1,4,5}, {0,1,5,6}};
    size_t V = 7;
    lda::model_definition def(3, V);
    lda::state state(def, alpha, beta, gamma, 2, docs, r);
    for(unsigned i = 0; i < 10; ++i){
        microscopes::kernels::lda_crp_gibbs(state, r);
    }
    state.validate_n_k_values();
    std::cout << "perplexity: " << state.perplexity() << std::endl;
}

static void
test_random_sequences(){
    sequence_random(0.2, 0.01, 0.5, 0);
    sequence_random(0.2, 0.01, 0.01, 6);
    sequence_random(0.2, 0.01, 0.5, 2);
    sequence_random(0.01, 0.001, 0.05, 13);
}


int main(void){
    test_random_sequences();
    std::cout << "test_random_sequences passed" << std::endl;
    return 0;
}