import pandas as pd


def get_actual_matches(series_qgrams, series_encoded_attr):
    """
    gets actual matches out of related columns of the datasets
    :param series_qgrams: pd.Series with the values of the plain text representation (i.e. set of bigrams) of a record
    :param series_encoded_attr: pd.Series with the values of the encoded representation (i.e. encoded bloom filter) of a record
    :return: a list of tuples which are the actual matches with their representation
    """
    series_qgrams = series_qgrams.apply(str)
    series_encoded_attr = series_encoded_attr.apply(str)
    return list(zip(series_qgrams, series_encoded_attr))

def import_data(filename, max_record_count, removed_fraction, random_state = None):
    """
    imports data from csv and possibly restricts it (by record count or removing random parts)
    :param filename (str): path of the csv file
    :param max_record_count (int): record count to be imported, if negative import from bottom, if 0 all data is imported
    :param removed_fraction (float): fraction of records to be removed (at random)
    :param random_state (int): random state for removing records
    :return: pd.DataFrame with all needed rows
    """
    plain_data = pd.read_csv(filename, na_filter=False)
    if max_record_count > 0:
        plain_data = plain_data.head(max_record_count)
    elif max_record_count < 0:
        plain_data = plain_data.tail(-max_record_count)
    plain_data = plain_data.sample(frac = 1 - removed_fraction, random_state = random_state)
    return plain_data
