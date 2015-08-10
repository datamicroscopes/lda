import os

from microscopes.lda import model, runner
from microscopes.lda.definition import model_definition
from microscopes.common.rng import rng
from microscopes.lda import utils

# Based on test_lda_reuters.py in ariddell's lda
# https://github.com/ariddell/lda/blob/57f721b05ffbdec5cb11c2533f72aa1f9e6ed12d/lda/tests/test_lda_reuters.py


class TestLDANewsReuters():

    def setUp(self):
        test_dir = os.path.dirname(__file__)
        reuters_ldac_fn = os.path.join(test_dir, 'data', 'reuters.ldac')
        with open(reuters_ldac_fn, 'r') as f:
            self.docs = utils.docs_from_ldac(f)

        self.V = utils.num_terms(self.docs)
        self.N = len(self.docs)

        self.defn = model_definition(self.N, self.V)
        self.prng = rng()
        self.latent = model.initialize(self.defn, self.docs, self.prng)
        self.r = runner.runner(self.defn, self.docs, self.latent)
        self.r.run(self.prng, 100)
        self.doc_topic = self.latent.document_distribution()


    def test_lda_news(self):
        assert len(self.doc_topic) == len(self.docs)
