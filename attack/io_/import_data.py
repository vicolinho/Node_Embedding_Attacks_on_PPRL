import pandas as pd


def get_actual_matches(series_qgrams, series_encoded_attr):
    """
    :param series_qgrams: pd.Series with the values of the plain text representation (i.e. set of bigrams) of a record
    :param series_encoded_attr: pd.Series with the values of the encoded representation (i.e. encoded bloom filter) of a record
    :return: a list of tuples which are the actual matches with their representation
    """
    series_qgrams = series_qgrams.apply(str)
    series_encoded_attr = series_encoded_attr.apply(str)
    return list(zip(series_qgrams, series_encoded_attr))

def import_data(filename, max_record_count, removed_fraction, random_state = None):
    """
    imports plain data from a csv file and adds necessary columns (i.e. set of bigrams)

    :param filename (str): path of the csv file
    :param max_record_count (int): record count to be imported, if negative import from bottom, if 0 all data is imported
    :param qgram_attributes (str):
    :param blk_attributes:
    :param encoded_attr:
    :param padding:
    :return: pd.DataFrame with all data
    """
    plain_data = pd.read_csv(filename, na_filter=False)
    if max_record_count > 0:
        plain_data = plain_data.head(max_record_count)
    elif max_record_count < 0:
        plain_data = plain_data.tail(-max_record_count)
    plain_data = plain_data.sample(frac = 1 - removed_fraction, random_state = random_state)
    return plain_data
    #return preprocessing.preprocess_plain_df(plain_data, qgram_attributes, blk_attributes, encoded_attr, padding)

def import_data_encoded(filename, max_record_count, encoded_attr):
    encoded_data = pd.read_csv(filename)
    if max_record_count > 0:
        encoded_data = encoded_data.head(max_record_count)
    elif max_record_count < 0:
        encoded_data = encoded_data.tail(-max_record_count)
    return encoded_data
    #return preprocessing.preprocess_encoded_df(encoded_data, encoded_attr)
