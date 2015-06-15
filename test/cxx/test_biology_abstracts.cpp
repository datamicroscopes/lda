#include <microscopes/lda/model.hpp>
#include <microscopes/lda/data.hpp>
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


static void
test_compare_biology_abstracts()
{
    rng_t r(5849343);

    std::cout << "creating model definition...";
    lda::model_definition def(5665, 4878);
    std::cout << " complete" << std::endl;

    std::cout << "initializing state...";
    lda::state state(def, 1, .5, 1, data::docs, r);
    std::cout << " complete" << std::endl;

    for(unsigned i = 0; i < 100; ++i){
        std::cout << "inference step: " << i << std::endl;
        state.inference();
        std::cout << "   K=" << state.usedDishes() << std::endl;
        std::cout << "   p=" << state.perplexity() << std::endl;
    }
    std::cout << "FINI!" << std::endl;
}

int main(void){
    test_compare_biology_abstracts();
}