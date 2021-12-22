import pandas as pd

from attack.adjust_sims import compute_number_of_qgrams


def node_features_plain(series_qgrams):
    # series of DataFrame containing q-grams
    qgram_counts = series_qgrams.apply(len)
    qgrams_strings = series_qgrams.apply(str)
    return pd.DataFrame({'qgrams':qgram_counts.to_numpy()}, index=qgrams_strings)

def node_features_encoded(series_bitarrays, series_encoded_attr ,bf_length, num_hash_f):
    # series of DataFrame containing q-grams
    bitarray_counts = series_bitarrays.apply(adjusted_number_of_qgrams, args=(bf_length, num_hash_f))
    id_strings = series_encoded_attr.apply(str)
    return pd.DataFrame({'qgrams': bitarray_counts.to_numpy()}, index=id_strings)

def adjusted_number_of_qgrams(bitarray, bf_length, num_hash_f):
    number_of_bits = bitarray.count(1)
    return compute_number_of_qgrams(bf_length, num_hash_f, number_of_bits)