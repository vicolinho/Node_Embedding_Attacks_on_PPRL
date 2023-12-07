import math

# functions to convert hamming weight of bloom filters into estimated qgram counts
# source: https://doi.org/10.1021/ci600526a

def compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits):
    """
    estimates qgram count with hamming weight bloom filter
    :param bf_length (int): length of bloom filter
    :param num_hash_f (int): number of hash functions used for bloom filter
    :param number_of_bits (int): hamming weight of bitarray
    :return: float: estimated qgram count
    """
    qgrams = -((bf_length) / num_hash_f) * \
                         math.log(1.0 - float(number_of_bits) / bf_length)
    return qgrams


def compute_number_of_common_qgrams(bf_length, num_hash_f, number_of_bits_a, number_of_bits_b, number_of_bits_a_plus_b):
    """
    estimates number of common qgrams between two records
    :param bf_length (int): length of bloom filter
    :param num_hash_f (int): number of hash functions used for bloom filter
    :param number_of_bits_a (int): hamming weight of first bitarray
    :param number_of_bits_b (int): hamming weight of second bitarray
    :param number_of_bits_a_plus_b (int): hamming weight of bitwise-OR of both bloom filters
    :return: float: estimated number of common qgrams between two records
    """
    qgrams = ((bf_length) / num_hash_f) * \
                         math.log((1.0 - float(number_of_bits_a_plus_b) / bf_length)/((1.0 - float(number_of_bits_a) / bf_length)*(1.0 - float(number_of_bits_b) / bf_length)))

    return max(qgrams,0)


def compute_adjusted_dice_from_bits(bitarray1, bitarray2, bf_length, num_hash_f):
    """
    computes adjusted dice similarity of pair of bloom filters
    :param bitarray1 (bitarray)
    :param bitarray2 (bitarray)
    :param bf_length (int): length of bloom filter
    :param num_hash_f (int): number of hash function encoded in bloom filter
    :return: float: adjusted dice similarity for bloom filters
    """
    bits_or = bitarray1 | bitarray2
    number_of_bits_a = bitarray1.count(1)
    number_of_bits_b = bitarray2.count(1)
    number_of_bits_a_plus_b = bits_or.count(1)
    a_ = compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits_a)
    b_ = compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits_b)
    a__times_b_ = compute_number_of_common_qgrams(bf_length, num_hash_f, number_of_bits_a, number_of_bits_b, number_of_bits_a_plus_b)
    if a_ + b_ > 0:
        return 2 * a__times_b_ / (a_ + b_)
    else:
        return 0.0