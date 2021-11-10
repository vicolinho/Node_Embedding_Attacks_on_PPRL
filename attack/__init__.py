import sys

from attack import blocking

import base64
import bitarray
from bitarray import bitarray
import pandas as pd
import networkx as nx
import numpy as np
from nltk import ngrams

data_plain_file = "../pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a.csv"
data_encoded_file = "../pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a_encoded_fn_ln.csv"

#primat

def record_sims_bf(df_encoded, attribute = 'base64_bf'):
    df_encoded = df_encoded.drop_duplicates(subset=attribute)
    df_encoded['bitarray'] = list(map(decode, df_encoded[attribute]))
    sim_dict = dict()
    array_key_bitarr = df_encoded.to_numpy()
    for idx1 in range(0,len(array_key_bitarr)):
        for idx2 in range(idx1+1, len(array_key_bitarr)):
            key = frozenset({array_key_bitarr[idx1,1],array_key_bitarr[idx2,1]})
            value = dice_sim_bfs(array_key_bitarr[idx1,3], array_key_bitarr[idx2,3])
            sim_dict[key] = value
    return sim_dict

def record_sims_plain_blk(blk_dict, attributes):
    sim_dict = dict()
    for key, value in blk_dict.items():
        sim_dict.update(record_sims_plain(value, attributes))
    return sim_dict

def record_sims_encoded_blk(blk_dict, attribute):
    sim_dict = dict()
    for key, value in blk_dict.items():
        sim_dict.update(record_sims_bf(value, attribute))
    return sim_dict

def record_sims_plain(df_plain, attributes):
    cols = []
    for attribute in attributes:
        cols.append(df_plain[attribute])
    df_plain['bigrams'] = list(map(get_bigrams, *cols))
    sim_dict = dict()
    array_key_bigram = df_plain['bigrams'].to_numpy()
    for idx1 in range(0,len(array_key_bigram)):
        for idx2 in range(idx1+1, len(array_key_bigram)):
            key = frozenset({array_key_bigram[idx1], array_key_bigram[idx2]})
            value = dice_sim_plain(array_key_bigram[idx1], array_key_bigram[idx2])
            if value != 1:
                sim_dict[key] = value
    return sim_dict

def get_bigrams(*args):
    s = set()
    for arg in args:
        s = s | (set(ngrams(arg,2)))
    return frozenset(s)

def dice_sim_bfs(bitarray1, bitarray2):
    bits_and = bitarray1 & bitarray2
    bit_count_sum = bitarray1.count(1) + bitarray2.count(1)
    if bit_count_sum == 0:
        return 0
    return 2 * bits_and.count(1) / bit_count_sum

def dice_sim_plain(bigrams1, bigrams2):
    intersection = bigrams1.intersection(bigrams2)
    if len(bigrams1) + len(bigrams2) != 0:
        return 2 * len(intersection) / (len(bigrams1) + len(bigrams2))
    else:
        return 0.0

def add_edges_from_sims(sim_array):
    list_edges = []
    return list_edges

def decode(base_string, length = 1024):
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

def graph_test():
    G = nx.Graph()
    G.add_nodes_from(range(0,5))
    G.add_node('1qeefwf')
    G.add_node(10)
    G.add_edges_from([(1,2, {'weight': 0.5}),(3,5),(1,4)])
    return G

def main():
    qgram_attributes = ['first_name', 'last_name']
    blk_attributes = ['first_name', 'last_name']
    encoded_attribute = 'base64_bf'
    plain_data = pd.read_csv(data_plain_file, na_filter=False)
    encoded_data = pd.read_csv(data_encoded_file)
    #plain_data, encoded_data = plain_data.head(1000), encoded_data.head(1000)
    plain_data['blocking'] = encoded_data['blocking'] = blocking.add_blocking_keys_to_dfs(plain_data, blk_attributes)
    blk_dict_plain = blocking.get_blocking_dict(plain_data)
    blk_dict_encoded = blocking.get_blocking_dict(encoded_data)
    sim_dict_plain = record_sims_plain_blk(blk_dict_plain, qgram_attributes)
    print(len(sim_dict_plain))
    sim_dict_encoded = record_sims_encoded_blk(blk_dict_encoded, encoded_attribute)
    # as requested in comment
    write_dict("../output_test/plain.txt", sim_dict_plain)
    write_dict("../output_test/encoded.txt", sim_dict_encoded)


def write_dict(filename, sim_dict):
    original_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("Length:", len(sim_dict))
        print(sim_dict)
        sys.stdout = original_stdout  # Reset the standard output to its original value


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print(end - start)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
