import unittest

import networkx as nx
import numpy as np

from page_rank import pageRankLinear, pageRankPower, randomWalk

class TestPageRank(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.matrix = np.matrix([[.0, .5, .0, .6],
                                 [.5, .0, .2, .0],
                                 [.0, .8, .0, .0],
                                 [.0, .7, .0, .0]])
        self.alpha = .9
        self.personalization = np.array([1., 0., 0., 0.])

        G = nx.from_numpy_array(self.matrix, create_using=nx.DiGraph)

        personalization = {node_id: value for node_id, value in enumerate(self.personalization.tolist())}
        scores = nx.pagerank(G, alpha=self.alpha, 
                             personalization=personalization)

        self.scores = list(scores.values())

    def check_vectors_almost_equal(self, l1, l2, precision=6):
        for i in range(len(l1)):
            self.assertAlmostEqual(l1[i], l2[i], places=precision)

    def test_page_rank_linear(self):
        scores = pageRankLinear(self.matrix, self.alpha, self.personalization)

        scores = scores.tolist()
        self.check_vectors_almost_equal(self.scores, scores, precision=6)

    def test_page_rank_power_method(self):
        scores = pageRankPower(self.matrix, self.alpha, self.personalization)

        scores = scores.tolist()
        self.check_vectors_almost_equal(self.scores, scores, precision=6)

    def test_page_rank_random_walk(self):
        scores = randomWalk(self.matrix, self.alpha, self.personalization)

        scores = scores.tolist()
        self.check_vectors_almost_equal(self.scores, scores, precision=6)

    def test_constistency(self):
        scores = pageRankLinear(self.matrix, self.alpha, self.personalization)
        scores_2 = pageRankPower(self.matrix, self.alpha, self.personalization)
        scores_3 = randomWalk(self.matrix, self.alpha, self.personalization)

        self.check_vectors_almost_equal(scores, scores_2)
        self.check_vectors_almost_equal(scores, scores_3)
        
if __name__ == '__main__':
    unittest.main(warnings='ignore')
