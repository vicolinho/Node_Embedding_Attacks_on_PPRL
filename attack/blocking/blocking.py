import random

from bitarray import bitarray

from attack.constants import BITARRAY, LSH_BLOCKING





def get_blocking_dict(df, blk_key_attr): #todo: additional needed attr
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

def get_dict_dataframes_by_blocking_keys(df, key_attr, sim_attr, bf_size, lsh_count, lsh_size):
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


