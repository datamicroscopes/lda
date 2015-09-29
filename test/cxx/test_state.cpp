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
using namespace microscopes::kernels::lda_crp;


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

// These tests are ported from
// https://github.com/shuyo/iir/blob/a6203a7523970a4807beba1ce3b9048a16013246/lda/test_hdplda2.py

static void
sequence4(double alpha, double beta, double gamma){
    rng_t r(5849343);
    std::vector< std::vector<size_t>> docs {{0,1,2,3}, {0,1,4,5}, {0,1,5,6}};
    size_t V = 7;
    lda::model_definition defn(3, V);
    lda::state state(defn, alpha, beta, gamma, 1, docs, r);
    auto Vbeta = V*beta;
    size_t k1 = state.create_dish();
    size_t k2 = state.create_dish();

    // Section 1
    size_t j = 0;
    size_t t1 = state.create_table(j, k1);
    size_t t2 = state.create_table(j, k2);
    state.add_table(j, t1, 0);
    state.add_table(j, t2, 1);
    state.add_table(j, t2, 2);
    state.add_table(j, t2, 3);

    j = 1;
    t1 = state.create_table(j, k1);
    t2 = state.create_table(j, k2);
    state.add_table(j, t2, 0);
    state.add_table(j, t2, 1);
    state.add_table(j, t1, 2);
    state.add_table(j, t2, 3);

    j = 2;
    t1 = state.create_table(j, k1);
    t2 = state.create_table(j, k2);
    state.add_table(j, t1, 0);
    state.add_table(j, t2, 1);
    state.add_table(j, t2, 2);
    state.add_table(j, t2, 3);

    // Section 2
    state.leave_from_dish(2, 1);
    state.seat_at_dish(2, 1, 2);

    state.remove_table(2, 0);
    state.add_table(2, 2, 0);

    state.leave_from_dish(0, 1);
    state.seat_at_dish(0, 1, 2);
    MICROSCOPES_CHECK(state.ntables() == 5, "state.ntables() is wrong in section 2");
    // return;
    MICROSCOPES_CHECK(state.m_k[1] == 1, "state.m_k[1] is wrong in section 2");
    MICROSCOPES_CHECK(state.m_k[2] == 4, "state.m_k[1] is wrong in section 2");
    state.leave_from_dish(1, 1);
    state.seat_at_dish(1, 1, 2);

    state.remove_table(2, 3);
    auto k_new = state.create_dish();
    MICROSCOPES_CHECK(k_new == 1, "k_new is wrong in section 2");
    auto t_new = state.create_table(j, k_new);
    MICROSCOPES_CHECK(t_new == 1, "t_new is wrong in section 2");
    state.add_table(2, 1, 3);
    // Section 3
    j = 0;
    size_t t = 1;
    state.leave_from_dish(j, t);

    auto p_k = calc_dish_posterior_t(state, j, t, r);
    float p0 = gamma / V;
    float p1 = 1 * beta / (V * beta + 1);
    float p2 = 4 * (beta + 2) / (Vbeta + 10);
    MICROSCOPES_CHECK(assertAlmostEqual(p_k[0], p0 / (p0 + p1 + p2)), "p_k[0] is wrong in section 3");
    MICROSCOPES_CHECK(assertAlmostEqual(p_k[1], p1 / (p0 + p1 + p2)), "p_k[1] is wrong in section 3");
    MICROSCOPES_CHECK(assertAlmostEqual(p_k[2], p2 / (p0 + p1 + p2)), "p_k[2] is wrong in section 3");

    state.seat_at_dish(j, t, 1);

    // Section 4
    t = 2;
    state.leave_from_dish(j, t);

    p_k = calc_dish_posterior_t(state, j, t, r);
    p0 = gamma * beta * beta * beta / (Vbeta * (Vbeta + 1) * (Vbeta + 2));
    p1 = 2 * (beta + 0) * beta * beta / ((Vbeta + 2) * (Vbeta + 3) * (Vbeta + 4));
    p2 = 3 * (beta + 2) * beta * beta / ((Vbeta + 7) * (Vbeta + 8) * (Vbeta + 9));
    MICROSCOPES_CHECK(assertAlmostEqual(p_k[0], p0 / (p0 + p1 + p2)), "p_k[0] is wrong in section 4");
    MICROSCOPES_CHECK(assertAlmostEqual(p_k[1], p1 / (p0 + p1 + p2)), "p_k[1] is wrong in section 4");
    MICROSCOPES_CHECK(assertAlmostEqual(p_k[2], p2 / (p0 + p1 + p2)), "p_k[2] is wrong in section 4");

}

static void
sequence3(double alpha, double beta, double gamma){
    rng_t r(5849343);
    std::vector< std::vector<size_t>> docs {{0,1,2,3}, {0,1,4,5}, {0,1,5,6}};
    size_t V = 7;
    lda::model_definition defn(3, V);
    lda::state state(defn, alpha, beta, gamma, 1, docs, r);

    size_t k1 = state.create_dish();
    size_t k2 = state.create_dish();

    // Section 1
    size_t j = 0;
    size_t t1 = state.create_table(j, k1);
    size_t t2 = state.create_table(j, k2);
    state.add_table(j, t1, 0);
    state.add_table(j, t2, 1);
    state.add_table(j, t1, 2);
    state.add_table(j, t1, 3);

    j = 1;
    t1 = state.create_table(j, k1);
    t2 = state.create_table(j, k2);
    state.add_table(j, t1, 0);
    state.add_table(j, t2, 1);
    state.add_table(j, t2, 2);
    state.add_table(j, t2, 3);

    j = 2;
    t1 = state.create_table(j, k1);
    t2 = state.create_table(j, k2);
    state.add_table(j, t1, 0);
    state.add_table(j, t2, 1);
    state.add_table(j, t2, 2);
    state.add_table(j, t2, 3);

    // Section 2
    std::vector<std::map<size_t, float>> phi = state.word_distribution();
    MICROSCOPES_CHECK(phi.size() == 2, "phi is wrong size");
    MICROSCOPES_CHECK(assertAlmostEqual(phi[0][0], (beta+3)/(V*beta+5)), "phi[0][0] is wrong in section 2");
    MICROSCOPES_CHECK(assertAlmostEqual(phi[0][2], (beta+1)/(V*beta+5)), "phi[0][2] is wrong in section 2");
    MICROSCOPES_CHECK(assertAlmostEqual(phi[0][3], (beta+1)/(V*beta+5)), "phi[0][3] is wrong in section 2");
    MICROSCOPES_CHECK(assertAlmostEqual(phi[1][1], (beta+3)/(V*beta+7)), "phi[1][1] is wrong in section 2");
    MICROSCOPES_CHECK(assertAlmostEqual(phi[1][4], (beta+1)/(V*beta+7)), "phi[1][4] is wrong in section 2");
    MICROSCOPES_CHECK(assertAlmostEqual(phi[1][5], (beta+2)/(V*beta+7)), "phi[1][5] is wrong in section 2");
    MICROSCOPES_CHECK(assertAlmostEqual(phi[1][6], (beta+1)/(V*beta+7)), "phi[1][6] is wrong in section 2");
    for(size_t v: {1, 4, 5, 6}){
        MICROSCOPES_CHECK(assertAlmostEqual(phi[0][v], (beta+0)/(V*beta+5)), "phi[0][v] is wrong");
    }
    for(size_t v: {0, 2, 3}){
        MICROSCOPES_CHECK(assertAlmostEqual(phi[1][v], (beta+0)/(V*beta+7)), "phi[1][v] is wrong");
    }

    // Section 3
    std::vector<std::vector<float>> theta = state.document_distribution();
    MICROSCOPES_CHECK(theta.size() == 3, "theta is wrong size");
    for(auto inner_theta: theta){
        MICROSCOPES_CHECK(inner_theta.size() == 3, "inner_theta is wrong size");
    }
    MICROSCOPES_CHECK(assertAlmostEqual(theta[0][0], (  alpha*gamma/(6+gamma))/(4+alpha)), "theta[0][0], (  alpha*gamma/(6+gamma))/(4+alpha) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(theta[0][1], (3+alpha*  3  /(6+gamma))/(4+alpha)), "theta[0][1], (3+alpha*  3  /(6+gamma))/(4+alpha) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(theta[0][2], (1+alpha*  3  /(6+gamma))/(4+alpha)), "theta[0][2], (1+alpha*  3  /(6+gamma))/(4+alpha) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(theta[1][0], (  alpha*gamma/(6+gamma))/(4+alpha)), "theta[1][0], (  alpha*gamma/(6+gamma))/(4+alpha) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(theta[1][1], (1+alpha*  3  /(6+gamma))/(4+alpha)), "theta[1][1], (1+alpha*  3  /(6+gamma))/(4+alpha) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(theta[1][2], (3+alpha*  3  /(6+gamma))/(4+alpha)), "theta[1][2], (3+alpha*  3  /(6+gamma))/(4+alpha) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(theta[2][0], (  alpha*gamma/(6+gamma))/(4+alpha)), "theta[2][0], (  alpha*gamma/(6+gamma))/(4+alpha) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(theta[2][1], (1+alpha*  3  /(6+gamma))/(4+alpha)), "theta[2][1], (1+alpha*  3  /(6+gamma))/(4+alpha) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(theta[2][2], (3+alpha*  3  /(6+gamma))/(4+alpha)), "theta[2][2], (3+alpha*  3  /(6+gamma))/(4+alpha) is wrong");

    // Section 4

    j = 0;
    size_t i = 0;
    size_t v = docs[j][i];

    state.remove_table(j, i);

    auto f_k = calc_f_k(state, v, r);
    MICROSCOPES_CHECK(f_k.size() == 3, "f_k is wrong size");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[1], (beta+2)/(V*beta+4)), "f_k[1] is wrong in section 4");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[2], (beta+0)/(V*beta+7)), "f_k[2] is wrong in section 4");

    auto p_t = calc_table_posterior(state, j, f_k, r);
    MICROSCOPES_CHECK(p_t.size() == 3, "p_t is wrong size");
    double p1 = 2*f_k[1];
    double p2 = 1*f_k[2];
    double p0 = alpha / (6+gamma) * (3*f_k[1] + 3*f_k[2] + gamma/V);
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[0], p0 / (p0+p1+p2)), "(p_t[0], p0 / (p0+p1+p2)) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[1], p1 / (p0+p1+p2)), "(p_t[1], p1 / (p0+p1+p2)) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[2], p2 / (p0+p1+p2)), "(p_t[2], p2 / (p0+p1+p2)) is wrong");

    state.add_table(j, 1, i);

    // Section 5
    j = 0;
    i = 1;
    v = docs[j][i];
    state.remove_table(j, i);
    MICROSCOPES_CHECK(state.using_t[j].size() == 2, "using_t[j] is wrong size");
    MICROSCOPES_CHECK(state.using_t[j][0] == 0, "using_t[j][0] is wrong");
    MICROSCOPES_CHECK(state.using_t[j][1] == 1, "using_t[j][1] is wrong");

    f_k = calc_f_k(state, v, r);
    MICROSCOPES_CHECK(f_k.size() == 3, "f_k is wrong size in section 5");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[1], (beta+0)/(V*beta+5)), "f_k[1] is wrong in section 5");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[2], (beta+2)/(V*beta+6)), "f_k[2] is wrong in section 5");

    p_t = calc_table_posterior(state, j, f_k, r);
    MICROSCOPES_CHECK(p_t.size() == 2, "p_t is wrong size in section 5");
    p1 = 3*f_k[1];
    p0 = alpha / (5+gamma) * (3*f_k[1] + 2*f_k[2] + gamma/V);
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[0], p0 / (p0+p1)), "(p_t[0], p0 / (p0+p1)) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[1], p1 / (p0+p1)), "(p_t[1], p1 / (p0+p1)) is wrong");
}

static void
sequence2(double alpha, double beta, double gamma){
    rng_t r(5849343);
    std::vector< std::vector<size_t>> docs {{0,1,2,3}, {0,1,4,5}, {0,1,5,6}};
    size_t V = 7;
    lda::model_definition defn(3, V);
    lda::state state(defn, alpha, beta, gamma, 1, docs, r);

    // assign all words to table 1 and all tables to dish 1
    size_t k_new = state.create_dish();
    MICROSCOPES_CHECK(k_new == 1, "k_new is wrong");
    for(size_t j: {0, 1, 2}){
        size_t t_new = state.create_table(j, k_new);
        MICROSCOPES_CHECK(t_new == 1, "j_new is wrong");
        for(size_t i: {0, 1, 2, 3}){
            state.add_table(j, t_new, i);
        }
    }
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_k.get(0), beta*V),
        "n_k[0] is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_k.get(1), beta*V+12),
        "n_k[1] is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[1].get(0), beta + 3),
        "n_kv[1].get(0) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[1].get(1), beta + 3),
        "n_kv[1].get(1) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[1].get(2), beta + 1),
        "n_kv[1].get(2) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[1].get(3), beta + 1),
        "n_kv[1].get(3) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[1].get(4), beta + 1),
        "n_kv[1].get(4) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[1].get(5), beta + 2),
        "n_kv[1].get(5) is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[1].get(6), beta + 1),
        "n_kv[1].get(6) is wrong");

    state.leave_from_dish(0, 1); // decreate m and m_k only
    MICROSCOPES_CHECK(state.ntables() == 2, "state.ntables() != 2");
    MICROSCOPES_CHECK(state.m_k[1] == 2, "state.m_k[1] != 2");

    state.seat_at_dish(0, 1, 1);
    MICROSCOPES_CHECK(state.ntables() == 3, "state.ntables() != 3");
    MICROSCOPES_CHECK(state.m_k[1] == 3, "state.m_k[1] != 3");
}


static void
sequence1(double alpha, double beta, double gamma){
    rng_t r(5849343);
    size_t V = 7;
    std::vector< std::vector<size_t>> docs {{0,1,2,3}, {0,1,4,5}, {0,1,5,6}};
    lda::model_definition defn(3, 7);
    lda::state state(defn, alpha, beta, gamma, 1, docs, r);

    // Section 1
    size_t j = 0;
    size_t i = 0;
    size_t v = docs[j][i];
    MICROSCOPES_CHECK(v == 0, "data wrong");

    std::vector<float> f_k = calc_f_k(state, v, r);
    std::vector<float> p_t = calc_table_posterior(state, j, f_k, r);
    MICROSCOPES_CHECK(assertSequenceEqual(p_t, std::vector<float> {1.}),
        "table posterior wrong");

    std::vector<float> p_k = calc_dish_posterior_w(state, f_k, r);
    MICROSCOPES_CHECK(p_k.size() == 1, "p_k has wrong number of elements");
    MICROSCOPES_CHECK(assertAlmostEqual(p_k[0], 1), "p_k has wrong element");

    size_t k_new = state.create_dish();
    MICROSCOPES_CHECK(k_new == 1, "incorrectly created new dish");
    size_t t_new = state.create_table(j, k_new);
    MICROSCOPES_CHECK(t_new == 1, "incorrectly created new table");
    MICROSCOPES_CHECK(state.dish_assignments()[j][t_new] == 1, "incorrectly created new table");

    MICROSCOPES_CHECK(
        assertSequenceEqual(state.using_t[j], std::vector<size_t> {0, 1}),
        "using_t[j] wrong after sitting at table");
    MICROSCOPES_CHECK(
        assertSequenceEqual(state.dishes_, std::vector<size_t> {0, 1}),
        "dishes_ wrong after sitting at table");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 0,
        "n_jt[j][t_new] wrong after sitting at table");

    state.add_table(j, t_new, i);
    MICROSCOPES_CHECK(state.table_assignments()[j][i] == 1,
        "table_assignments()[j][i] wrng after sitting at table");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 1,
        "n_jt[j][t_new] wrong after sitting at table");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[k_new].get(v), beta+1),
        "n_kv[k_new].get(v) wrong after sitting at table");

    // Section 2
    i = 1; // the existed table
    v = docs[j][i];
    MICROSCOPES_CHECK(v == 1, "we're not crazy");

    f_k = calc_f_k(state, v, r);
    MICROSCOPES_CHECK(f_k.size() == 2, "calc_f_k is wrong len when i = 1");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[1], beta / (V*beta+1)),
        "f_k[1] is wrong 2");
    p_t = calc_table_posterior(state, j, f_k, r);
    MICROSCOPES_CHECK(p_t.size() == 2, "p_t is wrong len when i = 1");
    double p0 = alpha / (1 + gamma) * (beta / (V * beta + 1) + gamma / V);
    double p1 = 1 * beta / (V * beta + 1);
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[0], p0 / (p0 + p1)),
        "p_t[0] is wrong"); // 0.10151692
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[1], p1 / (p0 + p1)),
        "p_t[1] is wrong 1"); // 0.89848308

    // Section 3
    t_new = 1;
    state.add_table(j, t_new, i);
    MICROSCOPES_CHECK(state.table_assignments()[j][i] == t_new, "state.table_assignments()[j][i] notset to t_new");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 2, "state.n_jt[j][t_new] incremented");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[k_new].get(v), beta+1),
        "n_kv[k_new].get(v) correct");

    // Section 4
    i = 2;
    v = docs[j][i];
    MICROSCOPES_CHECK(v == 2, "doc is corrupted :'''(");

    f_k = calc_f_k(state, v, r);
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[0], 0),
        "f_k[0] is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[1], (beta + 0) / (V*beta+2)),
        "f_k[1] is wrong 4");
    p_t = calc_table_posterior(state, j, f_k, r);
    MICROSCOPES_CHECK(p_t.size()==2, "p_t wrong size");
    p0 = alpha / (1 + gamma) * (beta / (V * beta + 2) + gamma / V);
    p1 = 2 * beta / (V * beta + 2);
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[0], p0 / (p0 + p1)),
        "p_t[0] is wrong"); // 0.05925473
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[1], p1 / (p0 + p1)),
        "p_t[1] is wrong 2"); // 0.94074527

    p_k = calc_dish_posterior_w(state, f_k, r);
    MICROSCOPES_CHECK(p_k.size() == 2, "p_k is wrong size in section 4");
    p0 = gamma / V;
    p1 = 1 * f_k[1];
    MICROSCOPES_CHECK(assertAlmostEqual(p_k[0], p0 / (p0 + p1)),
        "p_k[0] is wrong in section 4"); // 0.27835052
    MICROSCOPES_CHECK(assertAlmostEqual(p_k[1], p1 / (p0 + p1)),
        "p_k[1] is wrong in section 4"); // 0.72164948

    // Section 5
    k_new = 1; // TODO : calculate posterior of k

    t_new = state.create_table(j, k_new);
    MICROSCOPES_CHECK(t_new == 2, "t_new wrong in section 5");
    MICROSCOPES_CHECK(k_new == state.dish_assignments()[j][t_new], "k_new wrong in section 5");

    MICROSCOPES_CHECK(
        assertSequenceEqual(state.using_t[j], std::vector<size_t> {0, 1, 2}),
        "using_t[j] wrong after sitting at table in section 5");
    MICROSCOPES_CHECK(
        assertSequenceEqual(state.dishes_, std::vector<size_t> {0, 1}),
        "dishes_ wrong after sitting at table");

    state.add_table(j, t_new, i);
    MICROSCOPES_CHECK(state.table_assignments()[j][i] == t_new, "table_assignments() wrong in section 5");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 1, "n_jt wrong in section 5");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[k_new].get(v), beta+1),
        "n_kv[k_new].get(v) wrong in section 5");

    // Section 6
    i = 3;
    v = docs[j][i];
    MICROSCOPES_CHECK(v == 3, "doc is corrupted :(");

    f_k = calc_f_k(state, v, r);
    MICROSCOPES_CHECK(f_k.size() == 2, "f_k is wront length");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[0], 0),
        "f_k[0] is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[1], beta / (V*beta+3)),
        "f_k[1] is wrong 3");
    p_t = calc_table_posterior(state, j, f_k, r);
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
    state.add_table(j, t_new, i);
    MICROSCOPES_CHECK(state.table_assignments()[j][i] == t_new, "t_new is wrong");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 3, "n_jt[j][t_new] is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[k_new].get(v), beta + 1),
        "n_kv[k_new].get(v) isn't beta + 1");


    j = 1;
    i = 0;
    v = docs[j][i];
    MICROSCOPES_CHECK(v == 0, "docs are corrupted");

    f_k = calc_f_k(state, v, r);
    MICROSCOPES_CHECK(f_k.size() == 2, "f_k is the wrong size");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[0], 0), "f_k[0] is wrong");
    MICROSCOPES_CHECK(assertAlmostEqual(f_k[1], (beta + 1)/(V*beta+4)), "f_k[1] is wrong 1");

    p_t = calc_table_posterior(state, j, f_k, r);
    MICROSCOPES_CHECK(p_t.size(), "p_t is the wrong right size");
    MICROSCOPES_CHECK(assertAlmostEqual(p_t[0], 1), "p_T[0] is wrong");

    // add x_10 into a new table with dish 1
    k_new = 1;
    t_new = state.create_table(j, k_new);
    MICROSCOPES_CHECK(t_new == 1, "create_table failed to set t_new");

    MICROSCOPES_CHECK(assertSequenceEqual(state.using_t[j], std::vector<size_t> {0, 1}),
        "using_t[j] set incorrectly");
    MICROSCOPES_CHECK(assertSequenceEqual(state.dishes_, std::vector<size_t> {0, 1}),
        "dishes_ set incorrectly");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 0, "n_jt[j][t_new] set incorrectly");

    state.add_table(j, t_new, i);
    MICROSCOPES_CHECK(state.table_assignments()[j][i] == 1, "table_assignments()[j][i] set incorrectly");
    MICROSCOPES_CHECK(state.n_jt[j][t_new] == 1, "n_jt[j][t_new] set incorrectly");
    MICROSCOPES_CHECK(assertAlmostEqual(state.n_kv[k_new].get(v), beta+2), "n_kv[k_new].get(v)");
}

static void
test1(){
    sequence1(0.1, 0.1, 0.1);
}

static void
test2(){
    sequence1(0.2, 0.01, 0.5);
}

static void
test4(){
    sequence3(0.2, 0.01, 0.5);
}

static void
test5(){
    sequence4(0.2, 0.01, 0.5);
}

static void
test7(){
    sequence2(0.01, 0.001, 10);
}

static void
test8(){
    sequence2(0.01, 0.001, 0.05);
}


int main(void){
    test1();
    std::cout << "test1 passed" << std::endl;
    test2();
    std::cout << "test2 passed" << std::endl;
    test4();
    std::cout << "test4 passed" << std::endl;
    test5();
    std::cout << "test5 passed" << std::endl;
    test7();
    std::cout << "test7 passed" << std::endl;
    test8();
    std::cout << "test8 passed" << std::endl;
    return 0;

}