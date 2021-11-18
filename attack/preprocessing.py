import base64

import numpy as np
from bitarray import bitarray

BITARRAY = 'bitarray'


def decode(base_string, length=1024):
    bf_array = bitarray(length, endian='little')
    bf_array.setall(0)
    if isinstance(base_string, str):
        bf_string = base64.b64decode(base_string.strip())
        bf = list()
        for index, bit in enumerate(bf_string.strip()):
            bytes_little = bit.to_bytes(1, 'little')
            array = [access_bit(bytes_little, i) for i in range(len(bytes_little) * 8)]
            bf.extend(array)
        non_zero = np.nonzero(np.asarray(bf))
        for i in non_zero[0]:
            bf_array[int(i)] = 1
    return bf_array


def access_bit(data, num):
    base = int(num // 8)
    shift = int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def preprocess_encoded_df(df, encoded_attr):
    df = df.drop_duplicates(subset=encoded_attr)
    df[BITARRAY] = list(map(decode, df[encoded_attr]))
    return df


def preprocess_plain_df(df, lst_qgram_attr, lst_blocking_attr):
    duplicates_subset = lst_qgram_attr + lst_blocking_attr
    df = df.drop_duplicates(subset=duplicates_subset)
    return df
