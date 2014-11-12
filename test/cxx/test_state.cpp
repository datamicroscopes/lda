#include <string>
#include <vector>
#include <iostream>

#include <microscopes/common/variadic/dataview.hpp>
#include <microscopes/common/random_fwd.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/common/util.hpp>
#include <microscopes/lda/model.hpp>

using namespace std;
using namespace microscopes::io;
using namespace microscopes::common;
using namespace microscopes::lda;

typedef fixed_state::message_type message_type;

static inline string
to_crp_message(float alpha)
{
  CRP m;
  m.set_alpha(alpha);
  return util::protobuf_to_string(m);
}

static inline string
to_dirichlet_hp_message(const vector<float> &alphas)
{
  message_type m;
  for (auto a : alphas)
    m.add_alphas(a);
  return util::protobuf_to_string(m);
}

static inline size_t
arith_series(size_t n)
{
  return n*(n+1)/2;
}

static void
test1()
{
  const size_t N = 10, V = 100, K = 10;

  fixed_model_definition defn(N, V, K);
  unique_ptr< uint32_t[] > raw(new uint32_t[arith_series(N)]);

  vector<unsigned> ns;
  uint32_t *px = raw.get();
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < (i+1); j++, px++) {
      *px = j;
    }
    ns.push_back(i+1);
  }

  auto data = make_shared<variadic::row_major_dataview>(
      (const uint8_t *) raw.get(),
      ns,
      runtime_type(TYPE_U32));

  rng_t r;

  auto s = make_shared<fixed_state>(
      defn,
      to_dirichlet_hp_message(vector<float>(K, 1.)),
      to_dirichlet_hp_message(vector<float>(V, 1.)),
      *data,
      vector<vector<size_t>>(),
      r);

  fixed_model model(s, data);
  MICROSCOPES_CHECK(model.nentities() == N, "nentities()");
  MICROSCOPES_CHECK(model.ntopics() == K, "ntopics()");
  MICROSCOPES_CHECK(model.assignments().size() == N, "fail N");

  for (size_t i = 0; i < N; i++) {
    MICROSCOPES_CHECK(model.nterms(i) == (i + 1), "nterms(i)");
    MICROSCOPES_CHECK(model.assignments()[i].size() == (i + 1), "fail inner");
  }

  model_definition hdp_defn(N, V);

  auto hs = make_shared<state>(
      hdp_defn,
      to_crp_message(1.0),
      to_dirichlet_hp_message(vector<float>(V, 1.)),
      *data,
      10,
      vector<vector<size_t>>(),
      r);

  document_model doc_model(hs, data);
  MICROSCOPES_CHECK(doc_model.nentities() == N, "nentities()");
  MICROSCOPES_CHECK(doc_model.assignments().size() == N, "fail N");

  for (size_t i = 0; i < N; i++) {
    table_model t_model(hs, i);
    MICROSCOPES_CHECK(t_model.nentities() > 0, "ntables()");
  }
}

static void
VerifyTableSuffstats(
    const vector<map<size_t, vector<size_t>>> &table_suffstats,
    state &s)
{
  const size_t V = table_suffstats.front().begin()->second.size();
  for (size_t i = 0; i < table_suffstats.size(); i++) {
    for (const auto &p : table_suffstats[i]) {
      const auto &g = s.table_group(i, p.first);
      MICROSCOPES_CHECK((size_t)g.dim == V, "g.dim");
      for (size_t k = 0; k < V; k++) {
        MICROSCOPES_CHECK(g.counts[k] >= 0, "negative count");
        MICROSCOPES_CHECK((size_t)g.counts[k] == p.second[k], "counts !=");
      }
    }
  }
}

static void
VerifyDishSuffstats(
    const map<size_t, vector<size_t>> &dish_suffstats,
    state &s)
{
  const size_t V = dish_suffstats.begin()->second.size();
  for (const auto &p : dish_suffstats) {
    const auto &g = s.dish_group(p.first);
    MICROSCOPES_CHECK((size_t)g.dim == V, "g.dim");
    for (size_t k = 0; k < V; k++) {
      MICROSCOPES_CHECK(g.counts[k] >= 0, "negative count");
      MICROSCOPES_CHECK((size_t)g.counts[k] == p.second[k], "counts !=");
    }
  }
}

static void
test2()
{
  const size_t N = 10, V = 10;
  model_definition defn(N, V);

  unique_ptr< uint32_t[] > raw(new uint32_t[arith_series(N)]);

  vector<unsigned> ns;
  uint32_t *px = raw.get();
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < (i+1); j++, px++) {
      *px = j;
    }
    ns.push_back(i+1);
  }

  auto data = make_shared<variadic::row_major_dataview>(
      (const uint8_t *) raw.get(),
      ns,
      runtime_type(TYPE_U32));

  rng_t r;

  auto s = make_shared<state>(
      defn,
      to_crp_message(1.0),
      to_dirichlet_hp_message(vector<float>(V, 1.)),
      *data,
      10,
      vector<vector<size_t>>(),
      r);

  MICROSCOPES_CHECK(s->dish_alpha() == 1.0, "dish_alpha()");
  MICROSCOPES_CHECK(s->vocab_shared().dim == (size_t)V, "shared.dim");
  for (size_t i = 0; i < V; i++)
    MICROSCOPES_CHECK(s->vocab_shared().alphas[i] == 1., "shared.alphas[i]");

  // --- ensure empty dishes ---

  const auto empty_dishes = s->empty_dishes();
  if (empty_dishes.empty()) {
    s->create_dish(r);
  }

  // --- ensure empty tables ---

  for (size_t i = 0; i < N; i++) {
    const auto empty_tables = s->empty_tables(i);
    if (empty_tables.empty()) {
      s->create_table(i, r);
    }
  }

  // --- verify table suffstats ---

  vector< map<size_t, vector<size_t>> > table_suffstats;
  table_suffstats.resize(N);
  for (size_t i = 0; i < N; i++) {
    auto &m = table_suffstats[i];
    for (auto tid : s->tables(i))
      m[tid].resize(V);
  }

  auto table_assignments = s->table_assignments();
  MICROSCOPES_CHECK(table_assignments.size() == N, "table_assignments.size()");
  for (size_t i = 0; i < N; i++) {
    const size_t W = i + 1;
    MICROSCOPES_CHECK(table_assignments[i].size() == W, "table_assignments[i]");
    auto acc = data->get(i);
    for (size_t j = 0; j < W; j++) {
      MICROSCOPES_CHECK(table_assignments[i][j] != -1, "table_assignments[i][j] == -1");
      const size_t tid = table_assignments[i][j];
      auto it = table_suffstats[i].find(tid);
      MICROSCOPES_CHECK(it != table_suffstats[i].end(), "cant find tid");
      it->second[acc.get(j).get<uint32_t>()]++;
    }
  }

  VerifyTableSuffstats(table_suffstats, *s);

  // --- verify dish suffstats ---

  map<size_t, vector<size_t>> dish_suffstats;
  for (auto did : s->dishes()) {
    dish_suffstats[did].resize(V);
  }

  auto dish_assignments = s->assignments();
  MICROSCOPES_CHECK(dish_assignments.size() == N, "dish_assignments.size()");

  for (size_t i = 0; i < N; i++) {
    const size_t W = i + 1;
    auto acc = data->get(i);
    for (size_t j = 0; j < W; j++) {
      const size_t did = dish_assignments[i][j];
      auto it = dish_suffstats.find(did);
      MICROSCOPES_CHECK(it != dish_suffstats.end(), "cant find did");
      it->second[acc.get(j).get<uint32_t>()]++;
    }
  }

  VerifyDishSuffstats(dish_suffstats, *s);

  // --- test remove_value ---

  auto acc0 = data->get(0);
  for (size_t v = 0; v < acc0.n(); v++) {
    const size_t w = acc0.get(v).get<uint32_t>();
    const size_t did0 = dish_assignments[0][v];
    const size_t tid0 = table_assignments[0][v];
    const auto p = s->remove_value(0, v, acc0.get(v), r);
    MICROSCOPES_CHECK(did0 == p.first, "wrong did");
    MICROSCOPES_CHECK(tid0 == p.second, "wrong tid");
    table_suffstats[0][tid0][w] -= 1;
    dish_suffstats[did0][w] -= 1;
  }

  VerifyTableSuffstats(table_suffstats, *s);
  VerifyDishSuffstats(dish_suffstats, *s);

  // --- test score_value ---

  pair<vector<size_t>, vector<float>> scores;
  for (size_t v = 0; v < acc0.n(); v++) {
    s->inplace_score_value(scores, 0, v, acc0.get(v), r);
    // XXX(stephentu): needs some actual testing
  }

  // --- test add_value ---

  for (size_t v = 0; v < acc0.n(); v++) {
    const size_t w = acc0.get(v).get<uint32_t>();
    const size_t tid0 = 0;
    const size_t did0 = s->add_value(0, v, tid0, acc0.get(v), r);
    table_suffstats[0][tid0][w] += 1;
    dish_suffstats[did0][w] += 1;
  }

  VerifyTableSuffstats(table_suffstats, *s);
  VerifyDishSuffstats(dish_suffstats, *s);

}

static void
test3()
{
  const size_t N = 3, V = 10;
  model_definition defn(N, V);

  unique_ptr< uint32_t[] > raw(new uint32_t[arith_series(N)]);

  vector<unsigned> ns;
  uint32_t *px = raw.get();
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < (i+1); j++, px++) {
      *px = j;
    }
    ns.push_back(i+1);
  }

  auto data = make_shared<variadic::row_major_dataview>(
      (const uint8_t *) raw.get(),
      ns,
      runtime_type(TYPE_U32));

  rng_t r;

  auto s = make_shared<state>(
      defn,
      to_crp_message(1.0),
      to_dirichlet_hp_message(vector<float>(V, 1.)),
      *data,
      vector<vector<size_t>>({{1}, {2}, {5, 1}}),
      vector<vector<size_t>>({{0}, {0, 0}, {0, 1, 1}}),
      r);

  const auto a = s->assignments();
  const vector<vector<ssize_t>> check({{1}, {2, 2}, {5, 1, 1}});
  MICROSCOPES_CHECK(check == a, "assignments not correct");
}

int
main(void)
{
  test1();
  test2();
  test3();
  return 0;
}
