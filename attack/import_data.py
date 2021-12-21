import pandas as pd

from attack import preprocessing


def get_true_matches(series_qgrams, series_encoded_attr):
    series_qgrams = series_qgrams.apply(str)
    series_encoded_attr = series_encoded_attr.apply(str)
    return list(zip(series_qgrams, series_encoded_attr))

def import_data_plain(filename, max_record_count, qgram_attributes, blk_attributes):
    plain_data = pd.read_csv(filename, na_filter=False)
    if max_record_count > 0:
        plain_data = plain_data.head(max_record_count)
    return preprocessing.preprocess_plain_df(plain_data, qgram_attributes, blk_attributes)

def import_data_encoded(filename, max_record_count, encoded_attr):
    encoded_data = pd.read_csv(filename)
    if max_record_count > 0:
        encoded_data = encoded_data.head(max_record_count)
    return preprocessing.preprocess_encoded_df(encoded_data, encoded_attr)
