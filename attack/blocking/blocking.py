import random

from bitarray import bitarray

from attack.constants import BITARRAY, LSH_BLOCKING


def get_blocking_dict(df, blk_key_attr):
    """
    creates dict from pd.Dataframe and a attribute with the blocking key
    :param df (pd.Dataframe): records with calculated blocking key as attribute
    :param blk_key_attr: name of the calculated blocking key attribute
    :return: dict (key: bitarray, value: pd.Dataframe)
    """
    blk_dict = dict()
    grouped_blk_df = df.groupby(blk_key_attr)
    for blk_key in df[blk_key_attr].unique():
        blk_dict[blk_key] = grouped_blk_df.get_group(blk_key)
    return blk_dict

def lsh_blocking_key(attr, lst_positions):
    """
    generates lsh blocking key
    :param attr (bitarray): full bitarray of record
    :param lst_positions (list of int): list of bit position
    :return: bytes: lsh blocking key
    """
    lsh_key = bitarray(len(lst_positions), endian='little')
    lsh_key.setall(0)
    for i in range(0, len(lst_positions)):
        lsh_key[i] = attr[lst_positions[i]]
    return lsh_key.tobytes()

def choose_positions(lsh_count, lsh_size, bf_size):
    """

    :param lsh_count (int): how many lsh blocking keys to be generated
    :param lsh_size (int): length of lsh blocking key
    :param bf_size (int): bloom filter length
    :return: list of list of int: bit positions to generate lsh blocking key
    """
    lst_permutations = []
    for i in range(0, lsh_count):
        lst_positions = []
        for j in range(0, lsh_size):
            position = random.randrange(0,bf_size)
            if not position in lst_positions:
                lst_positions.append(position)
        lst_permutations.append(lst_positions)
    return lst_permutations

def add_lsh_blocking_columns(df, lst_permutations):
    """
    adds lsh blocking keys to dataframe
    :param df (df.Dataframe): records with bitarrays
    :param lst_permutations (list of list of int): bit positions to generate lsh blocking key
    :return: df.Dataframe with data for lsh blocking keys
    """
    col = df[BITARRAY].tolist()
    for i in range(0, len(lst_permutations)):
        df[LSH_BLOCKING+str(i)] = list(map(lsh_blocking_key, col, len(col) * [lst_permutations[i]]))
    return df

def get_dict_dataframes_by_blocking_keys(df, key_attr, sim_attr, bf_size, lsh_count, lsh_size):
    """
    :param df (pd.Dataframe):
    :param key_attr (str): attribute name of id of a record (frozenset with qgrams or base64 encoded bloom filter)
    :param sim_attr (str): attribute name of base for similarity calculation (frozenset with qgrams or bitarray bloom filter)
    :param bf_size (int): length of bloom filter
    :param lsh_count (int): number of lsh blocking keys
    :param lsh_size (int): length of lsh blocking key
    :return: list of dicts of pd.Dataframe: one dict includes dataframes with records matched to lsh blocking key
    """
    lst_dicts = []
    lst_permutations = choose_positions(lsh_count, lsh_size, bf_size)
    df = add_lsh_blocking_columns(df, lst_permutations)
    for i in range(0, lsh_count):
        blk_key_attr = LSH_BLOCKING + str(i)
        if key_attr == sim_attr:
            df_temp = df.loc[:, [blk_key_attr, key_attr]]
        else:
            df_temp = df.loc[:,[blk_key_attr, key_attr, sim_attr]]
        lst_dicts.append(get_blocking_dict(df_temp,blk_key_attr))
    return lst_dicts


