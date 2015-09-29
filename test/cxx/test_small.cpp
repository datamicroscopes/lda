#include <microscopes/lda/model.hpp>
#include <microscopes/lda/kernels.hpp>
#include <microscopes/lda/random_docs.hpp>
#include <microscopes/common/macros.hpp>
#include <microscopes/models/distributions.hpp>
#include <microscopes/common/random_fwd.hpp>

#include <random>
#include <iostream>
#include <time.h>

using namespace std;
using namespace distributions;
using namespace microscopes;
using namespace microscopes::common;

size_t
num_unique_words_in_docs(const std::vector< std::vector<size_t> > &docs){
    std::set<size_t> words;
    for(auto doc: docs)
        for(auto word: doc)
            words.insert(word);
    return words.size();
}



static void
test_small()
{
    rng_t r(time(NULL)); //

    std::cout << "creating model definition...";
    std::vector<std::vector<size_t>> docs = {{0, 1}, {2, 3}};
    lda::model_definition defn(docs.size(), num_unique_words_in_docs(docs));
    lda::state state(defn, .5, .01, .5, 1, docs, r);
    std::cout << " complete" << std::endl;

    for(unsigned i = 0; i < 100; ++i){
        std::cout << "inference step: " << i << std::endl;
        microscopes::kernels::lda_crp_gibbs(state, r);
        std::cout << "   p=" << state.perplexity() << std::endl;
        std::map<size_t, int> count;
        for(auto k_j: state.dish_assignments()){
            for(auto k: k_j){
                count[k]++;
            }
        }
        std::cout << "restaurants_ count  " << count << std::endl;
    }
    std::cout << "FINI!" << std::endl;
}

int main(void){
    test_small();
}