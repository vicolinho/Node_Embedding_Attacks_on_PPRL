import base64

import mmh3 as mmh3
from bitarray import bitarray
from nltk import ngrams

from attack.constants import BITARRAY, QGRAMS, NODE_COUNT
from attack.sim_graph.analysis import get_count_hash_func




def decode(byte_string, bf_length):
    bf = [access_bit(byte_string, i) for i in range(len(byte_string) * 8)]
    missing_bits = bf_length - len(bf)
    bf.extend(missing_bits * [0])
    return bitarray(bf)

def encode(bf):
    return base64.b64encode(bf.tobytes())

def access_bit(data, num):
    base = int(num // 8)
    shift = int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def preprocess_encoded_df(df, encoded_attr):
    bytes = df[encoded_attr].apply(base64.b64decode)
    bf_length = max(bytes.apply(len)) * 8
    df[BITARRAY] = bytes.apply(decode, args=(bf_length,))
    df = finalize_df_with_node_count(df, encoded_attr)
    return df


def preprocess_plain_df(plain_data, encoded_data, lst_qgram_attr, encoded_attr, padding, bf_length):
    df = add_qgrams_as_key(plain_data, lst_qgram_attr, padding)
    no_hash_func = get_count_hash_func(plain_data, encoded_data)
    df = add_bf_as_key(df, encoded_attr, bf_length, no_hash_func)
    df = finalize_df_with_node_count(df, QGRAMS)
    return df, no_hash_func


def add_qgrams_as_key(df, qgram_attributes, padding):
    cols = []
    for attribute in qgram_attributes:
        cols.append(df[attribute])
    if padding:
        df[QGRAMS] = list(map(get_bigrams_padding, *cols))
    else:
        df[QGRAMS] = list(map(get_bigrams, *cols))
    return df

def finalize_df_with_node_count(df, attribute):
    df[NODE_COUNT] = df.groupby(attribute)[attribute].transform("size")
    df = df.drop_duplicates(subset=attribute)
    return df

def add_bf_as_key(df, encoded_attr, bf_size, num_hash_func):
    df[BITARRAY] = qgrams_to_bfs(list(df[QGRAMS]), bf_size, num_hash_func)
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






def qgrams_to_bfs(qgrams_lists, bf_length, count_hash_func):
    bitarray_list = []
    seeds = list(range(0, count_hash_func))
    for qgrams_list in qgrams_lists:
        ba = bitarray(bf_length)
        ba.setall(0)
        for qgram in qgrams_list:
            qgram_string = ''.join(qgram)
            for seed in seeds:
                ba[mmh3.hash(qgram_string, seed) % bf_length] = 1
        bitarray_list.append(ba)
    return bitarray_list


def preprocess_dfs(encoded_data, plain_data, settings):
    encoded_data = preprocess_encoded_df(encoded_data, settings.encoded_attr)
    bf_length = len(encoded_data.loc[0, BITARRAY])
    plain_data, no_hash_func = preprocess_plain_df(plain_data, encoded_data, settings.qgram_attributes, settings.encoded_attr, settings.padding,
                                                   bf_length)
    return plain_data, encoded_data, no_hash_func, bf_length