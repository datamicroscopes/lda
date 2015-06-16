#include <microscopes/lda/model.hpp>
#include <microscopes/lda/random_docs.hpp>
#include <microscopes/common/macros.hpp>
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

namespace permutations{
    std::vector< std::vector<size_t>>
    rotate_labels(const std::vector< std::vector<size_t>> &docs, size_t vocab_size){
        std::vector< std::vector<size_t>> new_docs;
        new_docs.reserve(docs.size());
        for(auto doc: docs){
            std::vector<size_t> new_doc;
            new_doc.reserve(doc.size());
            for(auto word: doc){
                new_doc.push_back((word + 1) % vocab_size);
            }
            new_docs.push_back(new_doc);
        }
        return new_docs;
    }

    std::vector< std::vector<size_t>>
    rotate_docs(const std::vector< std::vector<size_t>> &docs, size_t vocab_size){
        std::vector< std::vector<size_t>> new_docs = docs;
        std::rotate(new_docs.begin(), new_docs.begin()+1, new_docs.end());
        return new_docs;
    }

    std::vector< std::vector<size_t>>
    shuffle_words(const std::vector< std::vector<size_t>> &docs, size_t vocab_size){
        std::vector< std::vector<size_t>> new_docs;
        new_docs.reserve(docs.size());
        for(auto doc: docs){
            std::vector<size_t> new_doc = doc;
            std::random_shuffle(new_doc.begin(), new_doc.end());
            new_docs.push_back(new_doc);
        }
        return new_docs;
    }
}


std::vector< std::vector<size_t>>
generate_random_docs(){
    return data::random_docs;
}

float
trial(const std::vector< std::vector<size_t>> &docs, size_t vocab_size,
      double alpha, double beta, double gamma, common::rng_t &r){
    size_t max_steps = 1000;
    lda::model_definition def(docs.size(), vocab_size);
    lda::state state(def, alpha, beta, gamma, docs, r);
    for(size_t i = 0; i < max_steps; i++){
        state.inference();
    }
    return state.perplexity();
}

void
test_permutations(){
    std::vector< std::vector<size_t>> docs = generate_random_docs();
    rng_t r(5849343);
    size_t vocab_size = 5;

    double p_baseline = trial(docs,
                              vocab_size, 1, .5, 1, r);
    double p_rotate_labels = trial(permutations::rotate_labels(docs, vocab_size),
                                   vocab_size, 1, .5, 1, r);
    double p_rotate_docs = trial(permutations::rotate_docs(docs, vocab_size),
                                 vocab_size, 1, .5, 1, r);
    double p_shuffle_words = trial(permutations::shuffle_words(docs, vocab_size),
                                   vocab_size, 1, .5, 1, r);

    MICROSCOPES_CHECK(assertAlmostEqual(p_baseline, p_rotate_labels, 0.1), "rotate labels test failed");
    MICROSCOPES_CHECK(assertAlmostEqual(p_baseline, p_rotate_docs, 0.1), "rotate docs test failed");
    MICROSCOPES_CHECK(assertAlmostEqual(p_baseline, p_shuffle_words, 0.1), "shuffle words test failed");
}


int main(void){
    test_permutations();
    std::cout << "permutations passed" << std::endl;
    return 0;

}