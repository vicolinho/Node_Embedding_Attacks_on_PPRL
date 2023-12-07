import math
from functools import reduce

from attack.constants import BITARRAY, QGRAMS


def get_count_hash_func(df_plain, df_encoded):
    """
    calculates predicted number of hashing functions used for bloom filter encoding based on samples
    :param df_plain: pd.DataFrame with plain text data (qgrams are needed)
    :param df_encoded: pd.DataFrame with encoded data (bitarrays are needed)
    :return: int: estimated number of hashing functions
    """
    bf_length = len(df_encoded.iloc[0][BITARRAY])
    df_plain = df_plain.sample(n = 1000, random_state=1) if len(df_plain) >= 1000 else df_plain
    df_encoded = df_encoded.sample(n = 1000, random_state=121) if len(df_encoded) >= 1000 else df_encoded
    ones = list(map(lambda ba: ba.count(1), df_encoded[BITARRAY]))
    ones_avg = reduce((lambda a, b: a+b), ones) / len(df_encoded)
    qgrams_num = list(map(lambda qgrams: len(qgrams), df_plain[QGRAMS]))
    qgrams_avg = reduce((lambda a, b: a+b), qgrams_num) / len(df_plain)
    return round(no_hash_func_avg(ones_avg, qgrams_avg, bf_length))


def no_hash_func_avg(bit_count, qgram_count, bf_length):
    """
    estimates number of hash function used for bloom filter encoding

    :param bit_count (float): average hamming weight  (b)
    :param qgram_count (float): average number of qgrams (q)
    :param bf_length (int): length of bloom filter (l)
    :return:

    # b = l * (1-(1- 1/l)^h*q) -> Maximum possible bit count * Probability a certain bit is set
    # logZ(-b/l + 1) = h*q  |:q
    # h = logZ(-b/l + 1) / q
    """
    z = 1 - 1 / bf_length
    return math.log(-bit_count/bf_length+1,z)/qgram_count