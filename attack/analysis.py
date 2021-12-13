import math
from functools import reduce

from attack import BITARRAY, QGRAMS


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
    bf_length = len(df_sample.iloc[0][BITARRAY]) # assuming all bit vectors have same length
    ones = list(map(lambda ba: ba.count(1), df_sample[BITARRAY]))
    ones_count = reduce((lambda a, b: a+b), ones)
    total = len(df_sample) * bf_length
    return ones_count / total


def false_negative_rate(df_encoded, lsh_size, lsh_count, dice_sim):
    freq_1s = get_frequency_1_bits(df_encoded)
    p_equals = p_equal_bits(freq_1s, dice_sim)
    return get_false_negatives(p_equals, lsh_size, lsh_count)


def get_num_hash_function(df_plain, df_encoded):
    BF_LENGTH = 1024
    # todo ensure that both have save length
    df_plain = df_plain.sample(frac=0.1, random_state=1)
    df_encoded = df_encoded.sample(frac=0.1, random_state=121)
    ones = list(map(lambda ba: ba.count(1), df_encoded[BITARRAY]))
    ones_avg = reduce((lambda a, b: a+b), ones) / len(df_encoded)
    qgrams_num = list(map(lambda qgrams: len(qgrams), df_plain[QGRAMS]))
    qgrams_avg = reduce((lambda a, b: a+b), qgrams_num) / len(df_plain)
    return get_number_hash_func(ones_avg, BF_LENGTH, qgrams_avg)


def get_number_hash_func(bit_count, bf_length, qgram_count):
    # b = l * (1-(1- 1/l)^h*q) -> Maximum possible bit count * Probability a certain bit is set
    # logZ(-b/l + 1) = h*q
    z = 1 - 1 / bf_length
    return math.log(-bit_count/bf_length+1,z)/qgram_count