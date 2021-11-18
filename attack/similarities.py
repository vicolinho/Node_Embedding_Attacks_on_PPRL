from nltk import ngrams
from itertools import combinations

from attack import BITARRAY

QGRAMS = 'qgrams'

# target sim_dict (for NetworkX): d = {node1: {node2: {"weight": 0.5}, ...}, ...}


def record_sims_bf(df, threshold=0.0):
    sim_dict = dict()
    pairs = list(combinations(df[BITARRAY], 2))
    for pair in pairs:
        key = frozenset({pair[0].tobytes(), pair[1].tobytes()})
        value = dice_sim_bfs(pair[0], pair[1])
        if value >= threshold:
            sim_dict[key] = value
    return sim_dict


def dice_sim_bfs(bitarray1, bitarray2):
    bits_and = bitarray1 & bitarray2
    bit_count_sum = bitarray1.count(1) + bitarray2.count(1)
    if bit_count_sum == 0:
        return 0
    return 2 * bits_and.count(1) / bit_count_sum


def record_sims_plain_blk(blk_dict, qgram_attributes, threshold = 0.0):
    sim_dict = dict()
    for key, value in blk_dict.items():
        sim_dict.update(record_sims_plain(value, qgram_attributes, threshold))
    return sim_dict


def record_sims_encoded_blk(blk_dict, threshold):
    sim_dict = dict()
    for key, value in blk_dict.items():
        sim_dict.update(record_sims_bf(value, threshold))
    return sim_dict


def record_sims_plain(df, qgram_attributes, threshold=0.0):
    cols = []
    for attribute in qgram_attributes:
        cols.append(df[attribute])
    df[QGRAMS] = list(map(get_bigrams, *cols))
    df = df.drop_duplicates(subset=QGRAMS)
    sim_dict = dict()
    pairs = list(combinations(df[QGRAMS], 2))
    for pair in pairs:
        key = frozenset({pair[0], pair[1]})
        value = dice_sim_plain(pair[0], pair[1])
        if value > threshold:
            sim_dict[key] = value
    return sim_dict


def get_bigrams(*args):
    s = set()
    for arg in args:
        s = s | (set(ngrams(arg, 2)))
    return frozenset(s)


def dice_sim_plain(bigrams1, bigrams2):
    intersection = bigrams1.intersection(bigrams2)
    if len(bigrams1) + len(bigrams2) != 0:
        return 2 * len(intersection) / (len(bigrams1) + len(bigrams2))
    else:
        return 0.0
