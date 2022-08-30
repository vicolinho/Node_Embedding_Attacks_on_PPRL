import base64

import numpy as np
from bitarray import bitarray
from nltk import ngrams

from attack import blocking

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

def encode(bf):
    return base64.b64encode(bf.tobytes())




def access_bit(data, num):
    base = int(num // 8)
    shift = int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def preprocess_encoded_df(df, encoded_attr):
    df = df.drop_duplicates(subset=encoded_attr)
    df[BITARRAY] = list(map(decode, df[encoded_attr]))
    return df


def preprocess_plain_df(df, lst_qgram_attr, lst_blocking_attr, encoded_attr, padding):
    duplicates_subset = lst_qgram_attr + lst_blocking_attr
    df = df.drop_duplicates(subset=duplicates_subset)
    df = add_qgrams_as_key(df, lst_qgram_attr, padding)
    df = add_bf_as_key(df, encoded_attr)
    return df


def add_qgrams_as_key(df, qgram_attributes, padding):
    cols = []
    for attribute in qgram_attributes:
        cols.append(df[attribute])
    if padding:
        df[QGRAMS] = list(map(get_bigrams_padding, *cols))
    else:
        df[QGRAMS] = list(map(get_bigrams, *cols))
    df = df.drop_duplicates(subset=QGRAMS)
    return df

def add_bf_as_key(df, encoded_attr):
    df[BITARRAY] = blocking.qgrams_to_bfs(list(df[QGRAMS]), 1024, 15)
    df[encoded_attr] = list(map(encode, df[BITARRAY]))
    return df

def get_bigrams(*args):
    s = set()
    for arg in args:
        s = s | (set(ngrams(arg, 2)))
    return frozenset(s)

def get_bigrams_padding(*args):
    s = set()
    for arg in args:
        s = s | (set(ngrams(arg, 2, pad_left=True, pad_right=True, left_pad_symbol='_', right_pad_symbol='_')))
    return frozenset(s)



QGRAMS = 'qgrams'