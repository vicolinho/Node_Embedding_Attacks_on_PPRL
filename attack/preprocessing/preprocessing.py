import base64

import mmh3 as mmh3
from bitarray import bitarray
from nltk import ngrams

from attack.constants import BITARRAY, QGRAMS, NODE_COUNT
from attack.sim_graph.analysis import get_count_hash_func




def decode(byte_string, bf_length):
    """
    decodes bloom filter
    :param byte_string (bytes)
    :param bf_length (int): length of bloom filter
    :return: bitarray: decoded bloom filter
    """
    bf = [access_bit(byte_string, i) for i in range(len(byte_string) * 8)]
    missing_bits = bf_length - len(bf)
    bf.extend(missing_bits * [0])
    return bitarray(bf)

def encode(bf):
    """
    encodes bloom filter (with base64)
    :param bf (bitarray): bloom filter
    :return: bytes: encoded bloom filter
    """
    return base64.b64encode(bf.tobytes())

def access_bit(data, num):
    """
    returns bit of byte string
    :param data (bytes): byte string
    :param num (int): number of bit to be accessed
    :return: int: value of bit
    """
    base = int(num // 8)
    shift = int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def preprocess_encoded_df(df, encoded_attr):
    """
    preprocesses encoded dataframe
    :param df (pd.DataFrame): encoded data
    :param encoded_attr (str): name of attribute with encoded bloom filter
    :return: pd.DataFrame: encoded data with decoded bloom filter and node count
    """
    bytes = df[encoded_attr].apply(base64.b64decode)
    bf_length = max(bytes.apply(len)) * 8
    df[BITARRAY] = bytes.apply(decode, args=(bf_length,))
    df = finalize_df_with_node_count(df, encoded_attr)
    return df


def preprocess_plain_df(plain_data, encoded_data, lst_qgram_attr, encoded_attr, padding, bf_length):
    """
    preprocesses plain data: calculates set of qgrams, adds node count, adds bitstring as key for lsh
    :param plain_data (pd.DataFrame)
    :param encoded_data (pd.DataFrame):
    :param lst_qgram_attr (list of str): lists attributes with are base of qgrams
    :param encoded_attr (str): attribute name of encoded bloom filter
    :param padding (bool): is padding used for qgram calculation
    :param bf_length (int): length of bloom filter
    :return: pd.DataFrame (plain data), int (number of hash functions)
    """
    df = add_qgrams_as_key(plain_data, lst_qgram_attr, padding)
    no_hash_func = get_count_hash_func(plain_data, encoded_data)
    df = add_bf_as_key(df, encoded_attr, bf_length, no_hash_func)
    df = finalize_df_with_node_count(df, QGRAMS)
    return df, no_hash_func


def add_qgrams_as_key(df, qgram_attributes, padding):
    """
    add qgrams column for plain dataframe
    :param df (pd.DataFrame): plain data
    :param qgram_attributes (list of str): list of base attributes for qgrams
    :param padding (bool): is padding to be used for qgrams
    :return: pd.DataFrame: plain data with new qgram attribute
    """
    cols = []
    for attribute in qgram_attributes:
        cols.append(df[attribute])
    if padding:
        df[QGRAMS] = list(map(get_bigrams_padding, *cols))
    else:
        df[QGRAMS] = list(map(get_bigrams, *cols))
    return df

def finalize_df_with_node_count(df, attribute):
    """
    adds node count and removes duplicates from data
    :param df (pd.DataFrame): plain or encoded data
    :param attribute (str): name of attribute representing the record
    :return: pd.DataFrame: data with node count and without duplicates
    """
    df[NODE_COUNT] = df.groupby(attribute)[attribute].transform("size")
    df = df.drop_duplicates(subset=attribute)
    return df

def add_bf_as_key(df, encoded_attr, bf_size, num_hash_func):
    """
    adds bloom filter as column (for plain data)
    :param df (pd.DataFrame): plain data
    :param encoded_attr (str): name of attribute with encoded representation
    :param bf_size (int): length of bloom filter
    :param num_hash_func (int): number of hash function to be used for bloom filter calculation
    :return: pd.DataFrame: plain data with columns for bloom filters used for lsh blocking
    """
    df[BITARRAY] = qgrams_to_bfs(list(df[QGRAMS]), bf_size, num_hash_func)
    df[encoded_attr] = list(map(encode, df[BITARRAY]))
    return df

def get_bigrams(*args):
    """
    calculates bigrams of strings without padding
    :param args (tuple of str): strings for qgram calculation
    :return: frozenset: qgram set
    """
    s = set()
    for arg in args:
        s = s | (set(ngrams(arg, 2)))
    return frozenset(s)

def get_bigrams_padding(*args):
    """
    calculates bigrams of strings with padding
    :param args (tuple of str): strings for qgram calculation
    :return: frozenset: qgram set
    """
    s = set()
    for arg in args:
        s = s | (set(ngrams(arg, 2, pad_left=True, pad_right=True, left_pad_symbol='_', right_pad_symbol='_')))
    return frozenset(s)


def qgrams_to_bfs(qgrams_lists, bf_length, count_hash_func):
    """
    converts list of qgram sets to bitarray list
    :param qgrams_lists (list of frozenset): list of sets of qgrams
    :param bf_length (int): length of bloom filter
    :param count_hash_func (int): number of hash function for bloom filter
    :return: list of bitarray
    """
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
    """
    preprocesses dataframes and calculates parameters for encoded data
    :param encoded_data (pd.DataFrame): encoded data imported from csv
    :param plain_data (pd.DataFrame): plain data imported from csv
    :param settings (Settings): set of parameters
    :return: pd.DataFrame, pd.DataFrame, int, int
    """
    encoded_data = preprocess_encoded_df(encoded_data, settings.encoded_attr)
    bf_length = len(encoded_data.iloc[0][BITARRAY])
    plain_data, no_hash_func = preprocess_plain_df(plain_data, encoded_data, settings.qgram_attributes, settings.encoded_attr, settings.padding,
                                                   bf_length)
    return plain_data, encoded_data, no_hash_func, bf_length