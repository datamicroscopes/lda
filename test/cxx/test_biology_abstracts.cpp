#include <microscopes/lda/model.hpp>
#include <microscopes/lda/data.hpp>
#include <microscopes/lda/random_docs.hpp>
#include <microscopes/common/macros.hpp>
#include <microscopes/common/recarray/dataview.hpp>
#include <microscopes/models/distributions.hpp>
#include <microscopes/common/random_fwd.hpp>
#include <microscopes/lda/kernels.hpp>

#include <random>
#include <iostream>
#include <time.h>

using namespace std;
using namespace distributions;
using namespace microscopes;
using namespace microscopes::common;
using namespace microscopes::common::recarray;


bool assertAlmostEqual(double a, double b, double epislon)
{
    return fabs(a - b) < epislon;
}

bool assertAlmostEqual(double a, double b)
{
    double EPSILON = 0.001;
    return assertAlmostEqual(a, b, EPSILON);
}


template<typename T>
bool assertSequenceEqual(const vector<T> &v1, const vector<T> &v2){
    if(v1.size() != v2.size()){
        return false;
    }
    for(size_t i = 0; i < v1.size(); i++){
        if(!assertAlmostEqual(v1[i], v2[i])){
            return false;
        }
    }
    return true;

}


size_t
num_unique_words_in_docs(const std::vector< std::vector<size_t> > &docs){
    std::set<size_t> words;
    for(auto doc: docs)
        for(auto word: doc)
            words.insert(word);
    return words.size();
}


static void
test_compare_biology_abstracts(size_t n_iterations)
{
    rng_t r(time(NULL));

    std::cout << "creating model definition...";
    size_t unique_words = num_unique_words_in_docs(data::docs);
    lda::model_definition def(data::docs.size(), unique_words);
    std::cout << data::docs.size() <<  " documents" << std::endl;
    std::cout << unique_words <<  " unique words in docs" << std::endl;
    std::cout << " complete" << std::endl;

    std::cout << "initializing state...";
    lda::state state(def, .1, .5, .1, data::docs, r);
    std::cout << " complete" << std::endl;

    for(unsigned i = 0; i < n_iterations; ++i){
        std::cout << "inference step: " << i << std::endl;
        microscopes::kernels::lda_crp_gibbs(state);
        std::cout << "   K=" << state.dishes_.size() - 1 << std::endl;
        std::cout << "   p=" << state.perplexity() << std::endl;
    }
    std::cout << "FINI!" << std::endl;
}

int main(int argc, char* argv[]){
    size_t n_iterations = SIZE_MAX;
    if(argc >= 2){
        n_iterations = std::stoi(argv[1]);
    }
    test_compare_biology_abstracts(n_iterations);
}