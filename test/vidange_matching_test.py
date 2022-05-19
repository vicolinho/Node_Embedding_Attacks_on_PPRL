import unittest

import numpy as np

from attack.node_matching import cos_sim_matrix_to_edges_vidange


class MyTestCase(unittest.TestCase):
    def test_something(self):
        cos_sims = self.initialize_cos_sim()
        threshold = 0.21
        w_cos, w_sim_conf, w_degr_conf = 5, 3, 2
        cos_sim_matrix_to_edges_vidange(cos_sims, threshold, w_cos, w_sim_conf, w_degr_conf)

    def initialize_cos_sim(self):
        cos_sims = np.array(8 * [8 * [0.2]])
        cos_sims[0, 1] = 0.81
        cos_sims[0, 2] = 0.7
        cos_sims[1, 0] = 0.9
        cos_sims[2, 2] = 1
        cos_sims[3, 1] = 0.87
        cos_sims[3, 2] = 0.94
        cos_sims[3, 3] = 0.87
        cos_sims[4, 4] = 0.92
        cos_sims[4, 5] = 0.89
        cos_sims[5, 4] = 0.99
        cos_sims[5, 6] = 0.78
        cos_sims[6, 7] = 0.91
        cos_sims[7, 7] = 0.95
        return cos_sims


# Eingangsmatrix cos_sim am besten mit Dummywerten (+ Gewichte etc.)
# Test auf Beispielzahlen aus dem Paper

if __name__ == '__main__':
    unittest.main()
