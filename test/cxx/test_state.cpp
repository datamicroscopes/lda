#include <string>
#include <vector>
#include <iostream>

#include <microscopes/common/variadic/dataview.hpp>
#include <microscopes/common/random_fwd.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/lda/model.hpp>

using namespace std;
using namespace microscopes::common;
using namespace microscopes::lda;

typedef fixed_state::message_type message_type;

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

  auto state = fixed_state::initialize(
      defn,
      to_dirichlet_hp_message(vector<float>(K, 1.)),
      to_dirichlet_hp_message(vector<float>(V, 1.)),
      *data,
      r);

  fixed_model model(state, data);
  MICROSCOPES_CHECK(model.nentities() == N, "nentities()");
  MICROSCOPES_CHECK(model.ntopics() == K, "ntopics()");
  MICROSCOPES_CHECK(model.assignments().size() == N, "fail N");

  for (size_t i = 0; i < N; i++) {
    MICROSCOPES_CHECK(model.nvariables(i) == (i + 1), "nvariables(i)");
    MICROSCOPES_CHECK(model.assignments()[i].size() == (i + 1), "fail inner");
  }

  return 0;
}
