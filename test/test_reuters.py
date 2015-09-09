import os
import numpy as np

from nose.tools import assert_almost_equal, assert_dict_equal
from nose.tools import assert_list_equal

from microscopes.lda import model, runner
from microscopes.lda.definition import model_definition
from microscopes.common.rng import rng
from microscopes.lda import utils

# Based on test_lda_reuters.py in ariddell's lda
# https://github.com/ariddell/lda/blob/57f721b05ffbdec5cb11c2533f72aa1f9e6ed12d/lda/tests/test_lda_reuters.py


class TestLDANewsReuters():

    @classmethod
    def _load_docs(cls):
        test_dir = os.path.dirname(__file__)
        reuters_ldac_fn = os.path.join(test_dir, 'data', 'reuters.ldac')
        with open(reuters_ldac_fn, 'r') as f:
            cls.docs = utils.docs_from_ldac(f)

        cls.V = utils.num_terms(cls.docs)
        cls.N = len(cls.docs)

    @classmethod
    def setup_class(cls):
        cls._load_docs()
        cls.niters = 100 if os.environ.get('TRAVIS') else 2

        cls.defn = model_definition(cls.N, cls.V)
        cls.seed = 12345
        cls.prng = rng(seed=cls.seed)
        cls.latent = model.initialize(cls.defn, cls.docs, cls.prng)
        cls.r = runner.runner(cls.defn, cls.docs, cls.latent)
        cls.r.run(cls.prng, cls.niters)
        cls.doc_topic = cls.latent.topic_distribution_by_document()

    def test_lda_news(self):
        assert len(self.doc_topic) == len(self.docs)

    def test_lda_monotone(self):
        # run additional iterations, verify improvement in log likelihood
        original_perplexity = self.latent.perplexity()
        self.r.run(self.prng, self.niters)
        print "self.latent.perplexity()", self.latent.perplexity()
        print "original_perplexity", original_perplexity
        assert self.latent.perplexity() < original_perplexity

    def test_lda_zero_iter(self):
        # compare to model with 0 iterations
        prng2 = rng(seed=54321)
        latent2 = model.initialize(self.defn, self.docs, prng2)
        assert latent2 is not None
        r2 = runner.runner(self.defn, self.docs, latent2)
        assert r2 is not None
        doc_topic2 = latent2.topic_distribution_by_document()
        assert doc_topic2 is not None
        assert latent2.perplexity() > self.latent.perplexity()

    def test_lda_random_seed(self):
        # ensure that randomness is contained in rng
        # by running model twice with same seed
        niters = 10

        # model 1
        prng1 = rng(seed=54321)
        latent1 = model.initialize(self.defn, self.docs, prng1)
        runner1 = runner.runner(self.defn, self.docs, latent1)
        runner1.run(prng1, niters)

        # model2
        prng2 = rng(seed=54321)
        latent2 = model.initialize(self.defn, self.docs, prng2)
        runner2 = runner.runner(self.defn, self.docs, latent2)
        runner2.run(prng2, niters)

        assert_list_equal(latent1.topic_distribution_by_document(),
                          latent2.topic_distribution_by_document())

        for d1, d2 in zip(latent1.word_distribution_by_topic(),
                          latent2.word_distribution_by_topic()):
            assert_dict_equal(d1, d2)

    def test_lda_attributes(self):
        assert np.array(self.doc_topic).shape == (self.N, self.latent.ntopics())
        assert len(self.latent.word_distribution_by_topic()) == self.latent.ntopics()
        for dist in self.latent.word_distribution_by_topic():
            assert len(dist) == self.V

        # check distributions sum to one
        for dist in self.latent.word_distribution_by_topic():
            assert_almost_equal(sum(dist.values()), 1)
        for dist in self.latent.topic_distribution_by_document():
            assert_almost_equal(sum(dist), 1)

    def test_lda_1transform_basic(self):
        n_docs = 3
        n_topics = self.latent.ntopics()
        docs_test = self.docs[0:n_docs]
        doc_topic_test = np.array(self.latent.predict(docs_test, self.prng))
        assert doc_topic_test.shape == (n_docs, n_topics)
        np.testing.assert_almost_equal(doc_topic_test.sum(axis=1), np.ones(n_docs))

        # one document
        docs_test = self.docs[0]
        doc_topic_test = np.array(self.latent.predict(docs_test, self.prng))
        doc_topic_test.shape = (1, n_topics)
        np.testing.assert_array_almost_equal(doc_topic_test.sum(axis=1), np.ones(1))
