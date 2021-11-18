import sys

from attack import blocking, preprocessing

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
    plain_data = pd.read_csv(DATA_PLAIN_FILE, na_filter=False)
    encoded_data = pd.read_csv(DATA_ENCODED_FILE)
    plain_data, encoded_data = plain_data.head(1000), encoded_data.head(1000)
    plain_data = preprocessing.preprocess_plain_df(plain_data, QGRAM_ATTRIBUTES, BLK_ATTRIBUTES)
    encoded_data = preprocessing.preprocess_encoded_df(encoded_data, ENCODED_ATTR)
    blk_dicts_plain = blocking.get_dict_dataframes_by_blocking_keys_plain(plain_data, BLK_ATTRIBUTES, blocking.soundex)
    blk_dicts_encoded = blocking.get_dict_dataframes_by_blocking_keys_encoded(encoded_data, BF_LENGTH, lsh_count=5, lsh_size=50)
    sim_dict_plain = record_sims_plain_blk(blk_dicts_plain, QGRAM_ATTRIBUTES)
    sim_dict_encoded = dict()
    for blk_dict_encoded in blk_dicts_encoded:
        sim_dict_encoded_temp = record_sims_encoded_blk(blk_dict_encoded, threshold=0.0)
        sim_dict_encoded = {**sim_dict_encoded, **sim_dict_encoded_temp}
    write_dict("output_test/plain.txt", sim_dict_plain)
    write_dict("output_test/encoded.txt", sim_dict_encoded)


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
