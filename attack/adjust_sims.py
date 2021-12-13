import math

# https://doi.org/10.1021/ci600526a
def compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits):
    qgrams = -((bf_length) / num_hash_f) * \
                         math.log(1.0 - float(number_of_bits) / bf_length)
    return qgrams


def compute_number_of_common_qgrams(bf_length, num_hash_f, number_of_bits_a, number_of_bits_b, number_of_bits_a_plus_b):
    qgrams = ((bf_length) / num_hash_f) * \
                         math.log((1.0 - float(number_of_bits_a_plus_b) / bf_length)/((1.0 - float(number_of_bits_a) / bf_length)*(1.0 - float(number_of_bits_b) / bf_length)))

    return max(qgrams,0)


def compute_number_of_united_qgrams(bf_length, num_hash_f, number_of_bits_a, number_of_bits_b, number_of_bits_a_plus_b):
    a = compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits_a_plus_b)
    b = -((bf_length) / num_hash_f) * \
                         math.log((1.0 - float(number_of_bits_a) / bf_length)*(1.0 - float(number_of_bits_b) / bf_length))
    return min(a,b)


def compute_real_dice_from_bits(bitarray1, bitarray2, bf_length, num_hash_f):
    bits_or = bitarray1 | bitarray2
    number_of_bits_a = bitarray1.count(1)
    number_of_bits_b = bitarray2.count(1)
    number_of_bits_a_plus_b = bits_or.count(1)
    a_ = compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits_a)
    b_ = compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits_b)
    a__times_b_ = compute_number_of_common_qgrams(bf_length, num_hash_f, number_of_bits_a, number_of_bits_b, number_of_bits_a_plus_b)
    a__plus_b_ = compute_number_of_united_qgrams(bf_length, num_hash_f, number_of_bits_a, number_of_bits_b, number_of_bits_a_plus_b)
    if a_ + b_ > 0:
        return 2 * a__times_b_ / (a_ + b_)
    else:
        return 0.0


def compute_number_of_bits(bf_length, num_hash_f, number_of_qgrams):
    est_num_bits = bf_length * math.pow(math.e, - (num_hash_f * number_of_qgrams) / bf_length)\
                    * (math.pow(math.e, (num_hash_f * number_of_qgrams) / bf_length) - 1)

    return est_num_bits