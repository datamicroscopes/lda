#pragma once

#include <microscopes/common/macros.hpp>
#include <microscopes/common/typedefs.hpp>
#include <microscopes/common/util.hpp>
#include <microscopes/common/variadic/dataview.hpp>
#include <microscopes/common/group_manager.hpp>
#include <distributions/models/dd.hpp>
#include <distributions/io/protobuf.hpp>

#include <memory>
#include <vector>

#include <eigen3/Eigen/Dense>

namespace microscopes {
namespace lda {

class fixed_model_definition {
public:
  /**
   * n - number of documents
   * v - vocabulary size
   * k - number of topics
   */
  fixed_model_definition(size_t n, size_t v, size_t k)
    : n_(n), v_(v), k_(k)
  {
    MICROSCOPES_DCHECK(n > 0, "no docs");
    MICROSCOPES_DCHECK(v > 0, "no terms");
    MICROSCOPES_DCHECK(k > 0, "no topics");
  }

  inline size_t n() const { return n_; }
  inline size_t v() const { return v_; }
  inline size_t k() const { return k_; }

private:
  size_t n_;
  size_t v_;
  size_t k_;
};

class fixed_state {
public:

  typedef distributions::protobuf::DirichletDiscrete_Shared message_type;

  fixed_state(const fixed_model_definition &def,
              const common::hyperparam_bag_t &topic_init,
              const common::hyperparam_bag_t &word_init,
              const common::variadic::dataview &data,
              const std::vector<std::vector<size_t>> &assignments,
              common::rng_t &rng)
  {
    message_type topic_m;
    common::util::protobuf_from_string(topic_m, topic_init);
    MICROSCOPES_DCHECK((size_t)topic_m.alphas_size() == def.k(),
        "topic mismatch");
    topic_alphas_.reserve(def.k());
    for (size_t i = 0; i < def.k(); i++)
      topic_alphas_.push_back(topic_m.alphas(i));

    sum_word_alphas_ = 0.;

    message_type word_m;
    common::util::protobuf_from_string(word_m, word_init);
    MICROSCOPES_DCHECK((size_t)word_m.alphas_size() == def.v(),
        "word mismatch");
    for (size_t i = 0; i < def.k(); i++) {
      word_alphas_.push_back(word_m.alphas(i));
      sum_word_alphas_ += word_alphas_.back();
    }

    doc_word_topics_.resize(def.n());
    doc_topic_counts_ = Eigen::MatrixXi::Zero(def.n(), def.k());
    topic_word_counts_ = Eigen::MatrixXi::Zero(def.k(), def.v());
    topic_counts_.resize(def.k());
    words_ = 0;

    MICROSCOPES_DCHECK(data.size() == def.n(), "data mismatch");
    std::vector<std::vector<size_t>> actual_assignments(assignments);
    if (actual_assignments.empty()) {
      actual_assignments.resize(def.n());
      std::uniform_int_distribution<unsigned> topic_dist(0, def.k() - 1);
      // XXX(stephentu): better initialization strategy
      for (size_t i = 0; i < def.n(); i++) {
        auto acc = data.get(i);
        actual_assignments[i].resize(acc.n());
        for (size_t j = 0; j < acc.n(); j++) {
          // pick a random topic blindly
          const size_t k = topic_dist(rng);
          actual_assignments[i][j] = k;
        }
      }
    }
    MICROSCOPES_DCHECK(actual_assignments.size() == def.n(),
        "invalid size assignment vector");

    for (size_t i = 0; i < def.n(); i++) {
      auto acc = data.get(i);
      doc_word_topics_[i].resize(acc.n());
      MICROSCOPES_DCHECK(actual_assignments[i].size() == acc.n(),
          "invalid size document assignment vector");
      for (size_t j = 0; j < acc.n(); j++) {
        const size_t w = acc.get(j).get<uint32_t>();
        MICROSCOPES_DCHECK(w < def.v(), "invalid entry");
        MICROSCOPES_DCHECK(actual_assignments[i][j] < def.k(), "invalid topic");
        const size_t k = actual_assignments[i][j];
        doc_word_topics_[i][j] = k;
        doc_topic_counts_(i, k) += 1;
        topic_word_counts_(k, w) += 1;
        topic_counts_[k] += 1;
        words_ += 1;
      }
    }
  }

  inline size_t nentities() const { return doc_word_topics_.size(); }
  inline size_t ntopics() const { return topic_counts_.size(); }
  inline size_t nwords() const { return word_alphas_.size(); }

  inline size_t
  nterms(size_t i) const
  {
    MICROSCOPES_DCHECK(i < nentities(), "invalid entity");
    return doc_word_topics_[i].size();
  }

  inline const std::vector<std::vector<ssize_t>> &
  assignments() const
  {
    return doc_word_topics_;
  }

  inline size_t
  remove_value(size_t eid,
               size_t vid,
               const common::value_accessor &value,
               common::rng_t &rng)
  {
    MICROSCOPES_DCHECK(eid < nentities(), "invalid entity");
    MICROSCOPES_DCHECK(vid < nterms(eid), "invalid vid");
    MICROSCOPES_DCHECK(
      doc_word_topics_[eid][vid] != -1, "(eid, vid) not assigned");

    const size_t w = value.get<uint32_t>();
    const size_t old_topic = doc_word_topics_[eid][vid];
    doc_word_topics_[eid][vid] = -1;
    MICROSCOPES_ASSERT(doc_topic_counts_(eid, old_topic));
    doc_topic_counts_(eid, old_topic) -= 1;
    MICROSCOPES_ASSERT(topic_word_counts_(old_topic, w));
    topic_word_counts_(old_topic, w) -= 1;

    return old_topic;
  }

  inline void
  add_value(size_t eid,
            size_t vid,
            size_t tid,
            const common::value_accessor &value,
            common::rng_t &rng)
  {
    MICROSCOPES_DCHECK(eid < nentities(), "invalid entity");
    MICROSCOPES_DCHECK(vid < nterms(eid), "invalid vid");
    MICROSCOPES_DCHECK(
      doc_word_topics_[eid][vid] == -1, "(eid, vid) assigned");
    MICROSCOPES_DCHECK(tid < ntopics(), "invalid tid");

    const size_t w = value.get<uint32_t>();
    doc_word_topics_[eid][vid] = tid;
    doc_topic_counts_(eid, tid) += 1;
    topic_word_counts_(tid, w) += 1;
  }

  inline std::pair<std::vector<size_t>, std::vector<float>>
  score_value(size_t eid,
              size_t vid,
              const common::value_accessor &value,
              common::rng_t &rng) const
  {
    std::pair<std::vector<size_t>, std::vector<float>> ret;
    ret.first.reserve(ntopics());
    ret.second.reserve(ntopics());
    inplace_score_value(ret, eid, vid, value, rng);
    return ret;
  }

  inline void
  inplace_score_value(
    std::pair<std::vector<size_t>, std::vector<float>> &scores,
    size_t eid,
    size_t vid,
    const common::value_accessor &value,
    common::rng_t &rng) const
  {
    scores.first.clear();
    scores.second.clear();

    using distributions::fast_log;

    const size_t w = value.get<uint32_t>();

    float term1_sum = 0.;
    for (size_t topic = 0; topic < ntopics(); topic++) {
      float sum = 0.;

      // P(topic | doc)
      const float term1 = doc_topic_counts_(eid, topic) + topic_alphas_[topic];
      term1_sum += term1;
      sum += fast_log(term1);

      // P(word | topic)
      const float term2 =
        (topic_word_counts_(topic, w) + word_alphas_[w]) /
        (words_ + sum_word_alphas_);

      sum += fast_log(term2);
      scores.first.push_back(topic);
      scores.second.push_back(sum);
    }

    const float lgnorm = fast_log(term1_sum);
    for (auto &s : scores.second)
      s -= lgnorm;
  }

private:

  /**
   * NOTE: we make a simplifying assumption that each topic
   * shares the same Dir(a) prior, and also each per-topic word
   * dist shares the same Dir(b) prior
   */

  // hyperparams for dirichlet prior on topic multinomial
  std::vector<float> topic_alphas_; // k

  // hyperparams for dirichlet prior on per-topic multinomial
  std::vector<float> word_alphas_; // v

  float sum_word_alphas_;

  // ij-th entry is the topic of the j-th word in the i-th document
  std::vector<std::vector<ssize_t>> doc_word_topics_; // n x variable

  // ij-th entry is the # of words of the j-th topic in the i-th document
  Eigen::MatrixXi doc_topic_counts_; // n x k

  // ij-th entry is the # of words of the j-th term in the i-th topic
  Eigen::MatrixXi topic_word_counts_; // k x v

  // i-th entry is # of words in the i-th topic
  std::vector<unsigned> topic_counts_; // k

  // the total # of words in every document
  unsigned words_;
};

class fixed_model {
public:
  fixed_model(const std::shared_ptr<fixed_state> &impl,
              const std::shared_ptr<common::variadic::dataview> &data)
    : impl_(impl), data_(data)
  {}

  inline size_t nentities() const { return impl_->nentities(); }
  inline size_t ntopics() const { return impl_->ntopics(); }
  inline size_t nterms(size_t i) const { return impl_->nterms(i); }
  inline size_t nwords() const { return impl_->nwords(); }
  inline const std::vector<std::vector<ssize_t>> & assignments() { return impl_->assignments(); }

  inline size_t
  remove_value(size_t eid, size_t vid, common::rng_t &rng)
  {
    const auto &value = data_->get(eid).get(vid);
    return impl_->remove_value(eid, vid, value, rng);
  }

  inline void
  add_value(size_t eid, size_t vid, size_t tid, common::rng_t &rng)
  {
    const auto &value = data_->get(eid).get(vid);
    impl_->add_value(eid, vid, tid, value, rng);
  }

  inline void
  inplace_score_value(
    std::pair<std::vector<size_t>, std::vector<float>> &scores,
    size_t eid,
    size_t vid,
    common::rng_t &rng) const
  {
    const auto &value = data_->get(eid).get(vid);
    impl_->inplace_score_value(scores, eid, vid, value, rng);
  }

private:
  std::shared_ptr<fixed_state> impl_;
  std::shared_ptr<common::variadic::dataview> data_;
};

class model_definition {
public:
  /**
   * n - number of documents
   * v - vocabulary size
   */
  model_definition(size_t n, size_t v)
    : n_(n), v_(v)
  {
    MICROSCOPES_DCHECK(n > 0, "no docs");
    MICROSCOPES_DCHECK(v > 0, "no terms");
  }

  inline size_t n() const { return n_; }
  inline size_t v() const { return v_; }

private:
  size_t n_;
  size_t v_;
};

/**
 * Implements the Chinese Restaurant Franchise (CRF) representation of
 * HDP-LDA. See:
 *
 *  Hierarchical Dirichlet Processes
 *  Teh et. al. 2006
 *  http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf
 *
 * This implementation is especially designed for collapsed gibbs sampling.
 */
class state {
public:

  static const size_t MaxVocabularySize = 0x10000;
  typedef distributions::DirichletDiscrete<MaxVocabularySize> DD;

  typedef io::CRP dish_message_type;
  typedef distributions::protobuf::DirichletDiscrete_Shared vocab_message_type;

  state(const model_definition &def,
        const common::hyperparam_bag_t &topic_init,
        const common::hyperparam_bag_t &word_init,
        const common::variadic::dataview &data,
        size_t k,
        const std::vector<std::vector<size_t>> &assignments,
        common::rng_t &rng)
  {
    MICROSCOPES_DCHECK(def.v() <= MaxVocabularySize, "vocab too large");
    MICROSCOPES_DCHECK(data.size() == def.n(), "data mismatch");
    MICROSCOPES_DCHECK(k > 0, "cannot start with zero dishes");

    dish_message_type topic_m;
    common::util::protobuf_from_string(topic_m, topic_init);
    MICROSCOPES_DCHECK(topic_m.alpha() > 0., "invalid alpha");
    alpha_ = topic_m.alpha();

    vocab_message_type word_m;
    common::util::protobuf_from_string(word_m, word_init);
    MICROSCOPES_DCHECK((size_t)word_m.alphas_size() == def.v(),
        "word mismatch");
    shared_.dim = def.v();
    for (size_t i = 0; i < def.v(); i++) {
      MICROSCOPES_DCHECK(word_m.alphas(i) > 0., "invalid alpha found");
      shared_.alphas[i] = word_m.alphas(i);
      shared_alphas_sum_ += word_m.alphas(i);
    }

    std::vector<std::vector<size_t>> actual_assignments(assignments);
    if (actual_assignments.empty()) {
      actual_assignments.resize(def.n());
      // XXX(stephentu):
      // arbitrarily start with 10 tables per document
      std::uniform_int_distribution<unsigned> topic_dist(0, 9);
      for (size_t i = 0; i < def.n(); i++) {
        auto acc = data.get(i);
        actual_assignments[i].resize(acc.n());
        for (size_t j = 0; j < acc.n(); j++) {
          // pick a random topic blindly
          const size_t k = topic_dist(rng);
          actual_assignments[i][j] = k;
        }
      }
    }

    MICROSCOPES_DCHECK(actual_assignments.size() == def.n(),
        "invalid size assignment vector");

    for (size_t i = 0; i < k; i++) {
      auto p = dishes_.create_group();
      p.second.group_.init(shared_, rng);
    }
    std::uniform_int_distribution<unsigned> dish_dist(0, k - 1);

    restaurants_.resize(def.n());
    for (size_t i = 0; i < def.n(); i++) {
      auto acc = data.get(i);
      MICROSCOPES_DCHECK(acc.n() > 0,
          "empty documents are not allowed");
      restaurants_[i] = common::group_manager<restaurant_suffstat_t>(acc.n());
      MICROSCOPES_DCHECK(actual_assignments[i].size() == acc.n(),
          "invalid size document assignment vector");
      const size_t ntables = *std::max_element(
          actual_assignments[i].begin(),
          actual_assignments[i].end()) + 1;
      for (size_t j = 0; j < ntables; j++) {
        auto p = restaurants_[i].create_group();
        MICROSCOPES_ASSERT(p.second.dish_ == -1);
        p.second.group_.init(shared_, rng);
        // XXX(stephentu): better dish assignment
        // randomly pick the dish
        p.second.dish_ = dish_dist(rng);
      }
      MICROSCOPES_ASSERT(restaurants_[i].ngroups() == ntables);
      for (size_t j = 0; j < acc.n(); j++) {
        const size_t w = acc.get(j).get<uint32_t>();
        MICROSCOPES_DCHECK(w < def.v(), "invalid entry");
        auto &table_ref = restaurants_[i].add_value(actual_assignments[i][j], j);
        table_ref.group_.add_value(shared_, w, rng);
        auto &dish = dishes_.group(table_ref.dish_);
        dish.group_.add_value(shared_, w, rng);
      }
    }
  }

  inline size_t nentities() const { return restaurants_.size(); }
  inline size_t ntopics() const { return dishes_.ngroups(); }
  inline size_t nwords() const { return shared_.dim; }

  inline size_t
  nterms(size_t eid) const
  {
    MICROSCOPES_DCHECK(eid < nentities(), "invalid eid");
    return restaurants_[eid].nentities();
  }

  inline size_t
  ntables(size_t eid) const
  {
    MICROSCOPES_DCHECK(eid < nentities(), "invalid eid");
    return restaurants_[eid].ngroups();
  }

  inline std::vector<size_t>
  tables(size_t eid) const
  {
    MICROSCOPES_DCHECK(eid < nentities(), "invalid eid");
    return restaurants_[eid].groups();
  }

  /**
   * NOTE: currently cannot distinguish between unassigned due
   * to lack of table assignment, or unassigned due to lack of
   * dish assignment. This shouldn't matter though, since we only
   * care about assignments when things are fully assigned.
   */
  inline std::vector<std::vector<ssize_t>>
  assignments() const
  {
    std::vector<std::vector<ssize_t>> ret;
    ret.resize(nentities());
    for (size_t i = 0; i < nentities(); i++) {
      auto &r = restaurants_[i];
      ret[i].resize(r.nentities());
      for (size_t j = 0; j < r.nentities(); j++) {
        const ssize_t tid = r.assignments()[j];
        if (tid == -1) {
          ret[i][j] = -1;
          continue;
        }
        ret[i][j] = r.group(tid).data_.dish_;
      }
    }
    return ret;
  }

  inline std::vector<std::vector<ssize_t>>
  table_assignments() const
  {
    std::vector<std::vector<ssize_t>> ret;
    ret.resize(nentities());
    for (size_t i = 0; i < nentities(); i++) {
      auto &r = restaurants_[i];
      ret[i].resize(r.nentities());
      for (size_t j = 0; j < r.nentities(); j++) {
        const ssize_t tid = r.assignments()[j];
        ret[i][j] = tid;
      }
    }
    return ret;
  }

  /**
   * Returns (top level, bottom level) ids removed
   */
  inline std::pair<size_t, size_t>
  remove_value(size_t eid,
               size_t vid,
               const common::value_accessor &value,
               common::rng_t &rng)
  {
    MICROSCOPES_DCHECK(eid < nentities(), "invalid eid");
    MICROSCOPES_DCHECK(vid < nterms(eid), "invalid vid");
    const size_t w = value.get<uint32_t>();
    MICROSCOPES_DCHECK(w < nwords(), "invalid w");
    auto ptable = restaurants_[eid].remove_value(vid);
    ptable.second.group_.remove_value(shared_, w, rng);
    auto &dish = dishes_.group(ptable.second.dish_);
    dish.group_.remove_value(shared_, w, rng);
    return std::make_pair(ptable.second.dish_, ptable.first);
  }

  inline size_t
  tablesize(size_t eid, size_t tid) const
  {
    MICROSCOPES_DCHECK(eid < nentities(), "invalid eid");
    return restaurants_[eid].groupsize(tid);
  }

  inline const std::set<size_t> &
  empty_tables(size_t eid) const
  {
    MICROSCOPES_DCHECK(eid < nentities(), "invalid eid");
    return restaurants_[eid].empty_groups();
  }

  inline void
  delete_table(size_t eid, size_t tid)
  {
    MICROSCOPES_DCHECK(eid < nentities(), "invalid eid");
    restaurants_[eid].delete_group(tid);
  }

  inline size_t
  dishsize(size_t did) const
  {
    return dishes_.group(did).group_.count_sum;
  }

  inline std::vector<size_t>
  dishes() const
  {
    return dishes_.groups();
  }

  inline std::set<size_t>
  empty_dishes() const
  {
    // XXX(stephentu): optimize this later
    // by merging the empty group code from group_manager to simple_group_manager
    // (and having group_manager be a thin wrapper around simple_group_manager
    // which also manages the CRP prior)
    std::set<size_t> ret;
    for (const auto &p : dishes_)
      if (!p.second.group_.count_sum)
        ret.insert(p.first);
    return ret;
  }

  inline void
  delete_dish(size_t did)
  {
    MICROSCOPES_DCHECK(dishsize(did) == 0, "dish not empty");
    dishes_.delete_group(did);
  }

  inline void
  inplace_score_value(
    std::pair<std::vector<size_t>, std::vector<float>> &scores,
    size_t eid,
    size_t vid,
    const common::value_accessor &value,
    common::rng_t &rng) const
  {
    using distributions::fast_log;

    MICROSCOPES_DCHECK(eid < nentities(), "invalid eid");
    MICROSCOPES_DCHECK(vid < nterms(eid), "invalid vid");

    scores.first.clear();
    scores.second.clear();

    const size_t w = value.get<uint32_t>();
    MICROSCOPES_DCHECK(w < nwords(), "invalid w");
    MICROSCOPES_DCHECK(restaurants_[eid].empty_groups().size(), "no empty tables");

    const size_t nempty_dishes = empty_dishes().size();
    MICROSCOPES_DCHECK(nempty_dishes > 0, "no empty dishes");

    // compute Eq. 31
    float sum = 0.;
    float pseudocounts = 0.;
    for (const auto &p : dishes_) {
      size_t pcount = p.second.group_.count_sum;
      if (!pcount)
        pcount = alpha_ / float(nempty_dishes);
      sum += float(pcount) *
             (shared_.alphas[w] + p.second.group_.counts[w]) /
             (shared_alphas_sum_ + p.second.group_.count_sum);
      pseudocounts += float(pcount);
    }
    const float lg_pr_word_new_table = fast_log(sum / pseudocounts);

    pseudocounts = 0.;
    for (const auto &p : restaurants_[eid]) {
      scores.first.push_back(p.first);
      const auto pcount = restaurants_[eid].pseudocount(p.first, p.second);
      const float lg_pcount = fast_log(pcount);
      if (!p.first) {
        scores.second.push_back(
          lg_pcount + lg_pr_word_new_table);
      } else {
        scores.second.push_back(
          lg_pcount + p.second.data_.group_.score_value(shared_, w, rng));
      }
      pseudocounts += pcount;
    }

    const float lgnorm = fast_log(pseudocounts);
    for (auto &s : scores.second)
      s -= lgnorm;
  }

  inline void
  add_value(size_t eid,
            size_t vid,
            size_t tid,
            const common::value_accessor &value,
            common::rng_t &rng)
  {
    using distributions::fast_log;

    MICROSCOPES_DCHECK(eid < nentities(), "invalid eid");
    MICROSCOPES_DCHECK(vid < nterms(eid), "invalid vid");

    const size_t w = value.get<uint32_t>();
    MICROSCOPES_DCHECK(w < nwords(), "invalid w");

    auto &restaurant = restaurants_[eid];
    auto &table = restaurant.add_value(tid, vid);
    table.group_.add_value(shared_, w, rng);

    if (table.dish_ != -1) {
      // easy case-- table is already assigned to a dish
      MICROSCOPES_ASSERT(dishsize(table.dish_));
      auto &dish = dishes_.group(table.dish_);
      dish.group_.add_value(shared_, w, rng);
    } else {
      // sample the dish for the table
      std::vector<float> scores;
      scores.reserve(ntopics());
      float pseudocounts = 0.;
      for (const auto &p : dishes_) {
        size_t pcount = p.second.group_.count_sum;
        if (!pcount)
          pcount = alpha_;
        const float likelihood = p.second.group_.score_value(shared_, w, rng);
        scores.push_back(fast_log(float(pcount)) + likelihood);
        pseudocounts += float(pcount);
      }

      const float lgnorm = fast_log(pseudocounts);
      for (auto &s : scores)
        s -= lgnorm;

      const size_t k = common::util::sample_discrete_log(scores, rng);
      table.dish_ = k;

      auto &dish = dishes_.group(table.dish_);
      dish.group_.add_value(shared_, w, rng);
    }
  }

  inline size_t
  remove_table(size_t eid,
               size_t tid,
               common::rng_t &rng)
  {
    MICROSCOPES_DCHECK(eid < nentities(), "invalid eid");
    auto &table = restaurants_[eid].group(tid);
    const ssize_t did = table.data_.dish_;
    MICROSCOPES_DCHECK(did != -1, "table has no dish assignment");
    auto &dish = dishes_.group(did);
    for (int i = 0; i < table.data_.group_.dim; i++) {
      const size_t cnt = table.data_.group_.counts[i];
      MICROSCOPES_ASSERT((size_t)dish.group_.counts[i] >= cnt);
      MICROSCOPES_ASSERT((size_t)dish.group_.count_sum >= cnt);
      dish.group_.counts[i] -= cnt;
      dish.group_.count_sum -= cnt;
    }

    table.data_.dish_ = -1;
    return did;
  }

  inline void
  inplace_score_table(
    std::pair<std::vector<size_t>, std::vector<float>> &scores,
    size_t eid,
    size_t tid,
    common::rng_t &rng) const
  {
    using distributions::fast_log;

    MICROSCOPES_DCHECK(eid < nentities(), "invalid eid");

    scores.first.clear();
    scores.second.clear();

    auto &table = restaurants_[eid].group(tid);
    MICROSCOPES_DCHECK(table.data_.dish_ == -1,
        "table assigned, cannot be scored");

    const size_t nempty_dishes = empty_dishes().size();
    MICROSCOPES_DCHECK(nempty_dishes > 0, "no empty dishes");

    // sparsify the table suffstats
    //
    // XXX(stephentu): this is a horrible hack for now--
    // we really want a sparse representation for the
    // dirichlet suff stats.
    std::map<size_t, size_t> table_suffstats;
    for (size_t i = 0; i < nwords(); i++) {
      const size_t cnt = table.data_.group_.counts[i];
      if (!cnt)
        continue;
      table_suffstats[i] = cnt;
    }

    float pseudocounts = 0.;
    for (const auto &p : dishes_) {
      size_t pcount = p.second.group_.count_sum;
      if (!pcount)
        pcount = alpha_ / float(nempty_dishes);
      float sum = fast_log(pcount);
      DD::Scorer scorer;
      scorer.init(shared_, p.second.group_, rng);
      for (const auto &pp : table_suffstats) {
        const float score = scorer.eval(shared_, pp.first, rng);
        sum += float(pp.second) * score;
      }
      scores.first.push_back(p.first);
      scores.second.push_back(sum);
    }

    const float lgnorm = fast_log(pseudocounts);
    for (auto &s : scores.second)
      s -= lgnorm;
  }

  inline size_t
  add_table(size_t eid,
            size_t tid,
            size_t did,
            common::rng_t &rng)
  {
    MICROSCOPES_DCHECK(eid < nentities(), "invalid eid");
    auto &table = restaurants_[eid].group(tid);
    MICROSCOPES_DCHECK(table.data_.dish_ == -1, "table assigned");

    auto &dish = dishes_.group(did);
    for (int i = 0; i < table.data_.group_.dim; i++) {
      const size_t cnt = table.data_.group_.counts[i];
      dish.group_.counts[i] += cnt;
      dish.group_.count_sum += cnt;
    }

    table.data_.dish_ = did;
  }

  // --- the methods exposed below are for testing ---

  inline float dish_alpha() const { return alpha_; }
  inline const DD::Shared & vocab_shared() const { return shared_; }

  inline const DD::Group &
  dish_group(size_t did) const
  {
    return dishes_.group(did).group_;
  }

  inline ssize_t
  table_dish(size_t eid, size_t tid) const
  {
    return restaurants_[eid].group(tid).data_.dish_;
  }

  inline const DD::Group &
  table_group(size_t eid, size_t tid) const
  {
    return restaurants_[eid].group(tid).data_.group_;
  }

private:

  struct dish_suffstat_t {
    DD::Group group_;
  };

  struct restaurant_suffstat_t {
    restaurant_suffstat_t() : group_(), dish_(-1) {}

    DD::Group group_;
    ssize_t dish_;
  };

  float alpha_; // hyperparam for the top level DP
  DD::Shared shared_; // hyperparam on the base measure of the individual DPs
  float shared_alphas_sum_; // normalization constant to turn the
                           // base measure dirichlet alphas into
                           // a probability distribution
  common::simple_group_manager<dish_suffstat_t> dishes_;
  std::vector<common::group_manager<restaurant_suffstat_t>> restaurants_;
};

class document_model {
public:
  document_model(const std::shared_ptr<state> &impl,
                 const std::shared_ptr<common::variadic::dataview> &data)
    : impl_(impl), data_(data)
  {}

  inline size_t nentities() const { return impl_->nentities(); }
  inline size_t ntopics() const { return impl_->ntopics(); }
  inline size_t nterms(size_t i) const { return impl_->nterms(i); }
  inline size_t nwords() const { return impl_->nwords(); }
  inline std::vector<std::vector<ssize_t>> assignments() const { return impl_->assignments(); }

  inline size_t dishsize(size_t did) const { return impl_->dishsize(did); }
  inline size_t tablesize(size_t eid, size_t tid) const { return impl_->tablesize(eid, tid); }

  inline void delete_dish(size_t did) { impl_->delete_dish(did); }
  inline void delete_table(size_t eid, size_t tid) { impl_->delete_table(eid, tid); }

  inline const std::set<size_t> & empty_tables(size_t eid) const { return impl_->empty_tables(eid); }
  inline std::set<size_t> empty_dishes() const { return impl_->empty_dishes(); }

  inline std::pair<size_t, size_t>
  remove_value(size_t eid, size_t vid, common::rng_t &rng)
  {
    const auto &value = data_->get(eid).get(vid);
    return impl_->remove_value(eid, vid, value, rng);
  }

  inline void
  inplace_score_value(
    std::pair<std::vector<size_t>, std::vector<float>> &scores,
    size_t eid,
    size_t vid,
    common::rng_t &rng) const
  {
    const auto &value = data_->get(eid).get(vid);
    impl_->inplace_score_value(scores, eid, vid, value, rng);
  }

  inline void
  add_value(size_t eid,
            size_t vid,
            size_t tid,
            common::rng_t &rng)
  {
    const auto &value = data_->get(eid).get(vid);
    impl_->add_value(eid, vid, tid, value, rng);
  }

private:
  std::shared_ptr<state> impl_;
  std::shared_ptr<common::variadic::dataview> data_;
};

/**
 * Presents an interface of tables being labelled 0 to ntables() - 1.
 */
class table_model {
public:
  table_model(const std::shared_ptr<state> &impl, size_t eid)
    : impl_(impl), eid_(eid), tid_mapping_(impl_->tables(eid))
  {
  }

  inline size_t ntopics() const { return impl_->ntopics(); }
  inline size_t ntables() const { return tid_mapping_.size(); }
  inline size_t dishsize(size_t did) const { return impl_->dishsize(did); }
  inline void delete_dish(size_t did) { impl_->delete_dish(did); }
  inline std::set<size_t> empty_dishes() const { return impl_->empty_dishes(); }

  inline size_t
  remove_table(size_t tid, common::rng_t &rng)
  {
    MICROSCOPES_DCHECK(tid < ntables(), "invalid tid");
    return impl_->remove_table(eid_, tid_mapping_[tid], rng);
  }

  inline void
  inplace_score_table(
    std::pair<std::vector<size_t>, std::vector<float>> &scores,
    size_t tid,
    common::rng_t &rng) const
  {
    MICROSCOPES_DCHECK(tid < ntables(), "invalid tid");
    impl_->inplace_score_table(scores, eid_, tid_mapping_[tid], rng);
  }

  inline void
  add_table(size_t tid, size_t did, common::rng_t &rng)
  {
    MICROSCOPES_DCHECK(tid < ntables(), "invalid tid");
    impl_->add_table(eid_, tid_mapping_[tid], did, rng);
  }

private:
  std::shared_ptr<state> impl_;
  size_t eid_;
  std::vector<size_t> tid_mapping_;
};

} // namespace lda
} // namespace microscopes
