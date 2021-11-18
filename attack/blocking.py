import random

from bitarray import bitarray
from pandas import DataFrame

from attack.preprocessing import BITARRAY

LSH_BLOCKING = 'lsh_blocking'
BLOCKING = 'blocking'

def blk_func(func, *args):
    key = str()
    for arg in args:
        key += func(arg)
    return key

sound_dict = {
    **dict.fromkeys(['a', 'e', 'i', 'o', 'u', 'y', 'h', 'w'], ''),
    **dict.fromkeys(['b', 'f', 'p', 'v'], '1'),
    **dict.fromkeys(['c', 'g', 'j', 'k', 'q', 's', 'x', 'z'], '2'),
    **dict.fromkeys(['d', 't'], '3'),
    **dict.fromkeys(['l'], '4'),
    **dict.fromkeys(['m', 'n'], '5'),
    **dict.fromkeys(['r'], '6')
  }

def soundex_core(attr):
    if len(attr) == 0:
        return '0000'
    elif len(attr) == 1:
        start, attr = attr[0], ''
    else:
        start, attr = attr[0], attr[1:]
    soundex_attr = ''
    last_number = -1
    for attr_char in attr:
        if not attr_char in sound_dict.keys():
            continue
        curr_number = sound_dict[attr_char]
        if curr_number != last_number:
            soundex_attr += curr_number
            last_number = curr_number if curr_number != '' else last_number
        if len(soundex_attr) >= 3:
            break
    while len(soundex_attr) < 3:
        soundex_attr += '0'
    soundex_attr = start + soundex_attr
    return soundex_attr

def soundex(attr_list):
    sdxcode = str()
    for attr in attr_list:
        sdxcode += soundex_core(attr)
    return sdxcode

def get_blocking_dict(df, blk_key_attr):
    blk_dict = dict()
    grouped_blk_df = df.groupby(blk_key_attr)
    for blk_key in df[blk_key_attr].unique():
        blk_dict[blk_key] = grouped_blk_df.get_group(blk_key)
    return blk_dict

def lsh_blocking_key(attr, lst_positions):
    lsh_key = bitarray(len(lst_positions), endian='little')
    lsh_key.setall(0)
    for i in range(0, len(lst_positions)):
        lsh_key[i] = attr[lst_positions[i]]
    return lsh_key.tobytes()

def choose_positions(count, lsh_size, bf_size):
    lst_permutations = []
    for i in range(0, count):
        lst_positions = []
        for j in range(0, lsh_size):
            position = random.randrange(0,bf_size)
            if not position in lst_positions:
                lst_positions.append(position)
        lst_permutations.append(lst_positions)
    return lst_permutations

def add_lsh_blocking_columns(df, lst_permutations):
    col = df[BITARRAY].tolist()
    for i in range(0, len(lst_permutations)):
        df[LSH_BLOCKING+str(i)] = list(map(lsh_blocking_key, col, len(col) * [lst_permutations[i]]))
    return df

def get_dict_dataframes_by_blocking_keys_encoded(df, bf_size, lsh_count, lsh_size):
    lst_dicts = []
    lst_permutations = choose_positions(lsh_count, lsh_size, bf_size)
    df = add_lsh_blocking_columns(df, lst_permutations)
    for i in range(0, lsh_count):
        lst_dicts.append(get_blocking_dict(df,LSH_BLOCKING + str(i)))
    return lst_dicts

def get_dict_dataframes_by_blocking_keys_plain(df, blk_attr_list, blk_func):
    df[BLOCKING] = add_blocking_keys_to_dfs(df, blk_attr_list, blk_func)
    return get_blocking_dict(df, BLOCKING)

def add_blocking_keys_to_dfs(df_all, blk_attr, blk_func):
    cols = DataFrame()
    for attribute in blk_attr:
        cols[attribute] = df_all[attribute]
    return cols.apply(func=blk_func, axis=1)