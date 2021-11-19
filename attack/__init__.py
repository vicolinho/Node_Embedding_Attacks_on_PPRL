import sys

from attack import blocking, preprocessing, sim_graph

import pandas as pd

from attack.preprocessing import BITARRAY
from attack.similarities import record_sims_plain_blk, record_sims_encoded_blk

DATA_PLAIN_FILE = "pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a.csv"
DATA_ENCODED_FILE = "pprl_datasets/ncvoter-20140619-temporal-balanced-ratio-1to1-a_encoded_fn_ln.csv"
QGRAM_ATTRIBUTES = ['first_name', 'last_name']
BLK_ATTRIBUTES = ['first_name', 'last_name']
ENCODED_ATTR = 'base64_bf'
BF_LENGTH = 1024

def main():
    graph_plain = create_sim_graph_plain(DATA_PLAIN_FILE, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES, blocking.soundex, 0.5, 180000)
    print(graph_plain)
    #graph_encoded = create_sim_graph_encoded(DATA_ENCODED_FILE, ENCODED_ATTR, BF_LENGTH, lsh_count = 3, lsh_size = 50, threshold = 0.0)
    #print(graph_encoded)

def create_sim_graph_encoded(file, encoded_attr, bf_length, lsh_count, lsh_size, threshold, max_record_count = -1):
    encoded_data = pd.read_csv(file)
    if max_record_count > 0:
        encoded_data = encoded_data.head(max_record_count)
    encoded_data = preprocessing.preprocess_encoded_df(encoded_data, encoded_attr)
    blk_dicts_encoded = blocking.get_dict_dataframes_by_blocking_keys_encoded(encoded_data, bf_length, lsh_count, lsh_size)
    sim_dict_encoded = dict()
    for blk_dict_encoded in blk_dicts_encoded:
        sim_dict_encoded_temp = record_sims_encoded_blk(blk_dict_encoded, threshold, encoded_attr)
        sim_dict_encoded = {**sim_dict_encoded, **sim_dict_encoded_temp}
    write_dict("output_test/encoded.txt", sim_dict_encoded)
    return sim_graph.convert_dict_to_graph(sim_dict_encoded)

def create_sim_graph_plain(file, qgram_attributes, blk_attributes, blk_func, threshold, max_record_count = -1):
    plain_data = pd.read_csv(file, na_filter=False)
    if max_record_count > 0:
        plain_data = plain_data.head(max_record_count)
    plain_data = preprocessing.preprocess_plain_df(plain_data, qgram_attributes, blk_attributes)
    blk_dicts_plain = blocking.get_dict_dataframes_by_blocking_keys_plain(plain_data, blk_attributes, blk_func)
    sim_dict_plain = record_sims_plain_blk(blk_dicts_plain, qgram_attributes, threshold)
    write_dict("output_test/plain.txt", sim_dict_plain)
    return sim_graph.convert_dict_to_graph(sim_dict_plain)


def write_dict(filename, sim_dict):
    original_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f
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
