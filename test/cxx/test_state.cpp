#include <microscopes/lda/model.hpp>
#include <microscopes/lda/data.hpp>
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

bool assertAlmostEqual(double a, double b)
{
    double EPSILON = 0.001;
    return fabs(a - b) < EPSILON;
}

template<typename T>
bool assertSequenceEqual(const vector<T> &v1, const vector<T> &v2){
    if(v1.size() != v2.size()){
        return false;
    }
    if (std::equal(v1.begin(), v1.end(), v2.begin())){
        return true;
    }
    return false;

}


static void
test_compare_shuyo()
{
    const size_t D = 28*28;
    rng_t r(5849343);

    vector<shared_ptr<models::model>> models;
    for (size_t i = 0; i < D; i++){
        models.emplace_back(make_shared<
            models::distributions_model<BetaBernoulli>>());
    }

    // std::cout << "loading data..."; std::cout.flush();

    std::cout << " complete" << std::endl;
    std::cout << "creating model definition...";
    lda::model_definition def(5665, 4878);
    std::cout << " complete" << std::endl;

    std::cout << "initializing state...";
    lda::state state(def, 1, .5, 1, data::docs, r);
    std::cout << " complete" << std::endl;
    std::cout << "inference!!" << std::endl;
    for(unsigned i = 0; i < 2000; ++i){

        state.inference();
        std::cout << "K=" << state.usedDishes() << std::endl;
        std::cout << "iter " << i << std::endl;
    }
    std::cout << "FINI!";
}

static void
test_create_model_def_and_state(){
    const size_t D = 28*28;
    rng_t r(5849343);

    vector<shared_ptr<models::model>> models;
    for (size_t i = 0; i < D; i++){
        models.emplace_back(make_shared<
            models::distributions_model<BetaBernoulli>>());
    }

    lda::model_definition def(2, 5);
    std::vector< std::vector<size_t>> docs {{0, 1, 2}, {2, 3, 4, 1}};
    lda::state state(def, 1, .5, 1, docs, r);
    for(unsigned i = 0; i < 20; ++i){
        state.inference();
    }
    std::cout << "test_create_model_def_and_state" << std::endl;
}

static void
sequence1(double alpha, double beta, double gamma){
    rng_t r(5849343);
    size_t V = 7;
    std::vector< std::vector<size_t>> docs {{0,1,2,3}, {0,1,4,5}, {0,1,5,6}};
    lda::model_definition def(3, 7);
    lda::state state(def, alpha, beta, gamma, docs, r);

    // Section 1
    size_t j = 0;
    size_t i = 0;
    size_t v = docs[j][i];
    MICROSCOPES_CHECK(v == 0, "data wrong");

    std::vector<float> f_k = state.calc_f_k(v);
    std::vector<float> p_t = state.calc_table_posterior(j, f_k);
    MICROSCOPES_CHECK(assertSequenceEqual(p_t, std::vector<float> {1.}),
        "table posterior wrong");

    std::vector<float> p_k = state.calc_dish_posterior_w(f_k);
    MICROSCOPES_CHECK(p_k.size() == 1, "p_k has wrong number of elements");
    MICROSCOPES_CHECK(assertAlmostEqual(p_k[0], 1), "p_k has wrong element");

    size_t k_new = state.add_new_dish();
    MICROSCOPES_CHECK(k_new == 1, "incorrectly created new dish");
    size_t t_new = state.add_new_table(j, k_new);
    MICROSCOPES_CHECK(t_new == 1, "incorrectly created new table");
    MICROSCOPES_CHECK(state.k_jt[j][t_new] == 1, "incorrectly created new table");

    MICROSCOPES_CHECK(
        assertSequenceEqual(state.using_t[j], std::vector<size_t> {0, 1}),
        "using_t[j] wrong after sitting at table");
    MICROSCOPES_CHECK(
        assertSequenceEqual(state.using_k, std::vector<size_t> {0, 1}),
        "using_k wrong after sitting at table");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 0,
        "n_jt[j][t_new] wrong after sitting at table");

    state.seat_at_table(j, i, t_new);
    MICROSCOPES_CHECK(state.t_ji[j][i] == 1,
        "t_ji[j][i] wrng after sitting at table");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 1,
        "n_jt[j][t_new] wrong after sitting at table");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[k_new][v], beta+1),
        "n_kv[k_new][v] wrong after sitting at table");

    // Section 2
    i = 1; // the existed table
    v = docs[j][i];
    MICROSCOPES_CHECK(v == 1, "we're not crazy");

    f_k = state.calc_f_k(v);
    MICROSCOPES_CHECK(f_k.size() == 2, "calc_f_k is wrong len when i = 1");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[1], beta / (V*beta+1)),
        "f_k[1] is wrong 2");
    p_t = state.calc_table_posterior(j, f_k);
    MICROSCOPES_CHECK(p_t.size() == 2, "p_t is wrong len when i = 1");
    double p0 = alpha / (1 + gamma) * (beta / (V * beta + 1) + gamma / V);
    double p1 = 1 * beta / (V * beta + 1);
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[0], p0 / (p0 + p1)),
        "p_t[0] is wrong"); // 0.10151692
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[1], p1 / (p0 + p1)),
        "p_t[1] is wrong 1"); // 0.89848308

    // Section 3
    t_new = 1;
    state.seat_at_table(j, i, t_new);
    MICROSCOPES_CHECK(state.t_ji[j][i] == t_new, "state.t_ji[j][i] notset to t_new");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 2, "state.n_jt[j][t_new] incremented");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[k_new][v], beta+1),
        "n_kv[k_new][v] correct");

    // Section 4
    i = 2;
    v = docs[j][i];
    MICROSCOPES_CHECK(v == 2, "doc is corrupted :'''(");

    f_k = state.calc_f_k(v);
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[0], 0),
        "f_k[0] is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[1], (beta + 0) / (V*beta+2)),
        "f_k[1] is wrong 4");
    p_t = state.calc_table_posterior(j, f_k);
    MICROSCOPES_CHECK(p_t.size()==2, "p_t wrong size");
    p0 = alpha / (1 + gamma) * (beta / (V * beta + 2) + gamma / V);
    p1 = 2 * beta / (V * beta + 2);
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[0], p0 / (p0 + p1)),
        "p_t[0] is wrong"); // 0.05925473
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[1], p1 / (p0 + p1)),
        "p_t[1] is wrong 2"); // 0.94074527

    p_k = state.calc_dish_posterior_w(f_k);
    MICROSCOPES_CHECK(p_k.size() == 2, "p_k is wrong size in section 4");
    p0 = gamma / V;
    p1 = 1 * f_k[1];
    MICROSCOPES_CHECK(assertAlmostEqual(p_k[0], p0 / (p0 + p1)),
        "p_k[0] is wrong in section 4"); // 0.27835052
    MICROSCOPES_CHECK(assertAlmostEqual(p_k[1], p1 / (p0 + p1)),
        "p_k[1] is wrong in section 4"); // 0.72164948

    // Section 5
    k_new = 1; // TODO : calculate posterior of k

    t_new = state.add_new_table(j, k_new);
    MICROSCOPES_CHECK(t_new == 2, "t_new wrong in section 5");
    MICROSCOPES_CHECK(k_new == state.k_jt[j][t_new], "k_new wrong in section 5");

    MICROSCOPES_CHECK(
        assertSequenceEqual(state.using_t[j], std::vector<size_t> {0, 1, 2}),
        "using_t[j] wrong after sitting at table in section 5");
    MICROSCOPES_CHECK(
        assertSequenceEqual(state.using_k, std::vector<size_t> {0, 1}),
        "using_k wrong after sitting at table");

    state.seat_at_table(j, i, t_new);
    MICROSCOPES_CHECK(state.t_ji[j][i] == t_new, "t_ji wrong in section 5");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 1, "n_jt wrong in section 5");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[k_new][v], beta+1),
        "n_kv[k_new][v] wrong in section 5");

    // Section 6
    i = 3;
    v = docs[j][i];
    MICROSCOPES_CHECK(v == 3, "doc is corrupted :(");

    f_k = state.calc_f_k(v);
    MICROSCOPES_CHECK(f_k.size() == 2, "f_k is wront length");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[0], 0),
        "f_k[0] is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[1], beta / (V*beta+3)),
        "f_k[1] is wrong 3");
    p_t = state.calc_table_posterior(j, f_k);
    MICROSCOPES_CHECK(p_t.size()==3, "p_t wrong size");
    p0 = alpha / (2 + gamma) * (2 * beta / (V * beta + 3) + gamma / V);
    p1 = 2 * beta / (V * beta + 3);
    double p2 = 1 * beta / (V * beta + 3);
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[0], p0 / (p0 + p1 + p2)),
        "p_t[0] is wrong in section 6"); // 0.03858731
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[1], p1 / (p0 + p1 + p2)),
        "p_t[1] is wrong in section 6"); // 0.64094179
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[2], p2 / (p0 + p1 + p2)),
        "p_t[2] is wrong in section 6"); // 0.3204709

    t_new = 1;
    state.seat_at_table(j, i, t_new);
    MICROSCOPES_CHECK(state.t_ji[j][i] == t_new, "t_new is wrong");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 3, "n_jt[j][t_new] is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[k_new][v], beta + 1),
        "n_kv[k_new][v] isn't beta + 1");


    j = 1;
    i = 0;
    v = docs[j][i];
    MICROSCOPES_CHECK(v == 0, "docs are corrupted");

    f_k = state.calc_f_k(v);
    MICROSCOPES_CHECK(f_k.size() == 2, "f_k is the wrong size");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[0], 0), "f_k[0] is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[1], (beta + 1)/(V*beta+4)), "f_k[1] is wrong 1");

    p_t = state.calc_table_posterior(j, f_k);
    MICROSCOPES_CHECK(p_t.size(), "p_t is the wrong right size");
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[0], 1), "p_T[0] is wrong");

    // add x_10 into a new table with dish 1
    k_new = 1;
    t_new = state.add_new_table(j, k_new);
    MICROSCOPES_CHECK(t_new == 1, "add_new_table failed to set t_new");

    MICROSCOPES_CHECK(assertSequenceEqual(state.using_t[j], std::vector<size_t> {0, 1}),
        "using_t[j] set incorrectly");
    MICROSCOPES_CHECK(assertSequenceEqual(state.using_k, std::vector<size_t> {0, 1}),
        "using_k set incorrectly");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 0, "n_jt[j][t_new] set incorrectly");

    state.seat_at_table(j, i, t_new);
    MICROSCOPES_CHECK(state.t_ji[j][i] == 1, "t_ji[j][i] set incorrectly");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 1, "n_jt[j][t_new] set incorrectly");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[k_new][v], beta+2), "n_kv[k_new][v]");
}

static void
test1(){
    sequence1(0.1, 0.1, 0.1);
}

static void
test2(){
    sequence1(0.2, 0.01, 0.5);
}



int main(void){
    // test_create_model_def_and_state();
    // test_compare_shuyo();
    test1();
    test2();
    return 0;

}