#pragma once

#include <microscopes/common/macros.hpp>
#include <microscopes/common/typedefs.hpp>
#include <microscopes/common/util.hpp>
#include <microscopes/common/variadic/dataview.hpp>
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

  fixed_state() = default;

  static std::shared_ptr<fixed_state>
  initialize(const fixed_model_definition &def,
             const common::hyperparam_bag_t &topic_init,
             const common::hyperparam_bag_t &word_init,
             const common::variadic::dataview &data,
             common::rng_t &rng)
  {
    auto px = std::make_shared<fixed_state>();

    message_type topic_m;
    common::util::protobuf_from_string(topic_m, topic_init);
    MICROSCOPES_DCHECK((size_t)topic_m.alphas_size() == def.k(),
        "topic mismatch");
    for (size_t i = 0; i < def.k(); i++)
      px->topic_alphas_.push_back(topic_m.alphas(i));

    px->sum_word_alphas_ = 0.;

    message_type word_m;
    common::util::protobuf_from_string(word_m, word_init);
    MICROSCOPES_DCHECK((size_t)word_m.alphas_size() == def.v(),
        "word mismatch");
    for (size_t i = 0; i < def.k(); i++) {
      px->word_alphas_.push_back(word_m.alphas(i));
      px->sum_word_alphas_ += px->word_alphas_.back();
    }

    px->doc_word_topics_.resize(def.n());
    px->doc_topic_counts_ = Eigen::MatrixXi::Zero(def.n(), def.k());
    px->topic_word_counts_ = Eigen::MatrixXi::Zero(def.k(), def.v());
    px->topic_counts_.resize(def.k());
    px->words_ = 0;

    std::uniform_int_distribution<unsigned> topic_dist(0, def.k() - 1);

    MICROSCOPES_DCHECK(data.size() == def.n(), "data mismatch");
    for (size_t i = 0; i < def.n(); i++) {
      auto acc = data.get(i);
      px->doc_word_topics_[i].resize(acc.n());
      for (size_t j = 0; j < acc.n(); j++) {
        const size_t w = acc.get(j).get<uint32_t>();
        MICROSCOPES_DCHECK(w < def.v(), "invalid entry");

        // pick a random topic blindly
        const size_t k = topic_dist(rng);
        px->doc_word_topics_[i][j] = k;

        px->doc_topic_counts_(i, k) += 1;
        px->topic_word_counts_(k, w) += 1;
        px->topic_counts_[k] += 1;
        px->words_ += 1;
      }
    }

    return px;
  }

  inline size_t nentities() const { return doc_word_topics_.size(); }
  inline size_t ntopics() const { return topic_counts_.size(); }

  inline size_t
  nvariables(size_t i) const
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
    MICROSCOPES_DCHECK(vid < nvariables(eid), "invalid vid");
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
    MICROSCOPES_DCHECK(vid < nvariables(eid), "invalid vid");
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
  inline size_t nvariables(size_t i) const { return impl_->nvariables(i); }
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

} // namespace lda
} // namespace microscopes
