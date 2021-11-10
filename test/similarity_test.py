import unittest

from pandas import DataFrame

import __init__
from bitarray import bitarray


class TestSimilarity(unittest.TestCase):
    def test_dice_sim_bfs(self):
        ba1 = bitarray('10011')
        ba2 = bitarray('11001')
        ba3 = bitarray('110110011')
        ba4 = bitarray('110110001')
        self.assertEqual(4 / 6, __init__.dice_sim_bfs(ba1, ba2))
        self.assertEqual(10 / 11, __init__.dice_sim_bfs(ba3, ba4))

    def test_record_sims_bf(self):
        #tests hier kaum möglich
        pass

    def test_record_sims_plain(self):
        data = {'first_name': ['annan','tim','anne','ethe'],'last_name': ['anhe','heth','li','tim']}
        df_plain = DataFrame(data=data)
        sim_dict = __init__.record_sims_plain(df_plain, ['first_name', 'last_name'])
        bigrams_annan = frozenset({('n','n'),('a','n'),('n','a'),('n','h'),('h','e')})
        bigrams_tim = frozenset({('t','h'),('e','t'), ('h','e'), ('i','m'), ('t','i')})
        bigrams_anne = frozenset({('n','e'),('n','n'),('l','i'),('a','n')})
        self.assertEqual(4/9, sim_dict[frozenset({bigrams_anne, bigrams_annan})])
        self.assertEqual(0.2, sim_dict[frozenset({bigrams_tim, bigrams_annan})])
        self.assertEqual(0, sim_dict[frozenset({bigrams_anne, bigrams_tim})])
        sim_test = {frozenset({bigrams_anne, bigrams_annan}): 4/9,
                    frozenset({bigrams_tim, bigrams_annan}): 0.2,
                    frozenset({bigrams_anne, bigrams_tim}): 0}
        self.assertEqual(sim_test, sim_dict)



if __name__ == '__main__':
    unittest.main()
