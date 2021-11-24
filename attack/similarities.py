import math
from functools import reduce

from nltk import ngrams
from itertools import combinations

from attack import BITARRAY

QGRAMS = 'qgrams'


# target sim_dict (for NetworkX): d = {node1: {node2: {"weight": 0.5}, ...}, ...}


def record_sims_bf(df, threshold, encoded_attr):
    sim_dict = dict()
    df = df.loc[:, [encoded_attr, BITARRAY]]
    records = df.to_records(index=False)
    pairs = list(combinations(records, 2))
    for pair in pairs:
        key = pair[0][0]
        sim = dice_sim_bfs(pair[0][1], pair[1][1])
        if sim >= threshold:
            if key not in sim_dict:
                sim_dict[key] = dict()
            value_dict = sim_dict[key]
            value_dict[pair[1][0]] = {"weight": sim}
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
        sim = dice_sim_plain(pair[0], pair[1])
        if sim >= threshold:
            if pair[0] not in sim_dict:
                sim_dict[pair[0]] = dict()
            value_dict = sim_dict[pair[0]]
            value_dict[pair[1]] = {"weight": sim}
    return sim_dict


def dice_sim_bfs(bitarray1, bitarray2):
    bits_and = bitarray1 & bitarray2
    bit_count_sum = bitarray1.count(1) + bitarray2.count(1)
    if bit_count_sum == 0:
        return 0
    return 2 * bits_and.count(1) / bit_count_sum


def record_sims_plain_blk(blk_dict, qgram_attributes, threshold=0.0):
    sim_dict = dict()
    for key, value in blk_dict.items():
        sim_dict.update(record_sims_plain(value, qgram_attributes, threshold))
    return sim_dict


def record_sims_encoded_blk(blk_dict, threshold, encoded_attr):
    sim_dict = dict()
    for key, value in blk_dict.items():
        sim_dict.update(record_sims_bf(value, threshold, encoded_attr))
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

# https://doi.org/10.1021/ci600526a
def compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits):
    qgrams = -((bf_length) / num_hash_f) * \
                         math.log(1.0 - float(number_of_bits) / bf_length)
    return qgrams

for bf_length in [1024]:
    for num_hash_f in range(1,4):
        for number_of_bits in [10, 20, 5]:
            print(bf_length, num_hash_f, number_of_bits, compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits))

def compute_number_of_common_qgrams(bf_length, num_hash_f, number_of_bits_a, number_of_bits_b, number_of_bits_a_plus_b):
    qgrams = -((bf_length) / num_hash_f) * \
                         math.log((1.0 - float(number_of_bits_a_plus_b) / bf_length)/((1.0 - float(number_of_bits_a) / bf_length)*(1.0 - float(number_of_bits_b) / bf_length)))

    return max(qgrams,0)

def compute_number_of_united_qgrams(bf_length, num_hash_f, number_of_bits_a, number_of_bits_b, number_of_bits_a_plus_b):
    a = compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits_a_plus_b)
    b = -((bf_length) / num_hash_f) * \
                         math.log((1.0 - float(number_of_bits_a) / bf_length)*(1.0 - float(number_of_bits_b) / bf_length))
    return min(a,b)

def compute_real_dice_from_bits(bf_length, num_hash_f, number_of_bits_a, number_of_bits_b, number_of_bits_a_plus_b):
    a_ = compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits_a)
    b_ = compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits_b)
    a__plus_b_ = compute_number_of_united_qgrams(bf_length, num_hash_f, number_of_bits_a, number_of_bits_b, number_of_bits_a_plus_b)
    if a_ + b_ > 0:
        return 2 * a__plus_b_ / (a_ + b_)
    else:
        return 0.0

def compute_number_of_bits(bf_length, num_hash_f, number_of_qgrams):
    est_num_bits = bf_length * math.pow(math.e, - (num_hash_f * number_of_qgrams) / bf_length)\
                    * (math.pow(math.e, (num_hash_f * number_of_qgrams) / bf_length) - 1)

    return est_num_bits

def p_equal_bits(freq_1s, dice_sim):
    p_11 = freq_1s * dice_sim
    p_10 = p_01 = 0.5 * p_11 * ((1 - dice_sim) / dice_sim)
    p_00 = 1 - p_11 - p_01 - p_10
    p_equals = p_11 + p_00
    return p_equals

def get_false_negatives(p_equals, lsh_size, lsh_count):
    p_same_bit_vector = math.pow(p_equals, lsh_size)
    return math.pow(1-p_same_bit_vector, lsh_count)

def get_frequency_1_bits(df_encoded):
    df_sample = df_encoded.sample(frac=0.01, random_state=1)
    bf_length = len(df_sample.iloc[0][BITARRAY]) #assuming all bit vectors have same length
    ones = list(map(lambda ba: ba.count(1), df_sample[BITARRAY]))
    ones_count = reduce((lambda a, b: a+b), ones)
    total = len(df_sample) * bf_length
    return ones_count / total

def false_negative_rate(df_encoded, lsh_size, lsh_count, dice_sim):
    freq_1s = get_frequency_1_bits(df_encoded)
    p_equals = p_equal_bits(freq_1s, dice_sim)
    return get_false_negatives(p_equals, lsh_size, lsh_count)