#include <string>
#include <vector>
#include <iostream>

#include <microscopes/common/variadic/dataview.hpp>
#include <microscopes/common/random_fwd.hpp>
#include <microscopes/common/assert.hpp>
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

int
main(void)
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
    MICROSCOPES_CHECK(t_model.ntables() > 0, "ntables()");
  }

  return 0;
}
