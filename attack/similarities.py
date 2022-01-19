import pandas as pd
from itertools import combinations

from pandas import DataFrame
from stellargraph.globalvar import SOURCE, TARGET, WEIGHT

from preprocessing import QGRAMS
import sim_graph
from preprocessing import BITARRAY, add_qgrams_as_key
from adjust_sims import compute_real_dice_from_bits


def edges_df_from_blk_plain(blk_dict, qgram_attributes, threshold, id):
    return edges_df_from_blk(blk_dict, edges_df_from_blk_element_plain, qgram_attributes, threshold, id)

def edges_df_from_blk_bf(blk_dict, encoded_attr, threshold, id):
    return edges_df_from_blk(blk_dict, edges_df_from_blk_element_bf, encoded_attr, threshold, id)

def edges_df_from_blk_bf_adjusted(blk_dict, threshold, encoded_attr, bf_length, num_of_hash_func, id):
    return edges_df_from_blk(blk_dict, edges_df_from_blk_element_bf_adjusted, encoded_attr, threshold, id, bf_length=bf_length, num_of_hash_func=num_of_hash_func)


def edges_df_from_blk(blk_dict, sim_func, sim_attr, threshold, id, **kwargs):
    df = DataFrame()
    for key, value in blk_dict.items():
        df_temp = sim_func(value, sim_attr, threshold, id, **kwargs)
        df = pd.concat([df, df_temp])
    return df


def edges_df_from_blk_element_plain(df, qgram_attributes, threshold, id):
    # df = add_qgrams_as_key(df, qgram_attributes)
    return edges_df_from_blk_element(df, threshold, id=id, node_attribute=QGRAMS, sim_attribute=QGRAMS, sim_func=dice_sim_plain)


def edges_df_from_blk_element_bf(df, encoded_attr, threshold, id):
    return edges_df_from_blk_element(df, threshold, id=id, node_attribute=encoded_attr, sim_attribute=BITARRAY, sim_func=dice_sim_bfs)

def edges_df_from_blk_element_bf_adjusted(df, encoded_attr, threshold, id, bf_length, num_of_hash_func):
    return edges_df_from_blk_element(df, threshold, id=id, node_attribute=encoded_attr, sim_attribute=BITARRAY, sim_func=compute_real_dice_from_bits, num_hash_f=num_of_hash_func, bf_length=bf_length)

def edges_df_from_blk_element(df, threshold, node_attribute, sim_attribute, sim_func, id, **kwargs):
    arr_first, arr_second, arr_sims = [], [], []
    if node_attribute == sim_attribute:
        df = df.loc[:, [node_attribute]]
    else:
        df = df.loc[:, [node_attribute, sim_attribute]]
    records = df.to_records(index=False)  # needed for using combinations function
    pairs = list(combinations(records, 2))
    for pair in pairs:
        if node_attribute != sim_attribute:
            sim = sim_func(pair[0][1], pair[1][1], **kwargs)
        else:
            sim = sim_func(pair[0][0], pair[1][0], **kwargs)
        if sim >= threshold:
            arr_first.append(sim_graph.adjust_node_id(pair[0][0], id))
            arr_second.append(sim_graph.adjust_node_id(pair[1][0], id))
            arr_sims.append(sim)
    d = {SOURCE: arr_first, TARGET: arr_second , WEIGHT: arr_sims}
    return pd.DataFrame(d)

def get_comp_rec_pairs(df, node_attribute, sim_attribute): # must integrate both functions which are redundant
    if node_attribute == sim_attribute:
        df = df.loc[:, [node_attribute]]
    else:
        df = df.loc[:, [node_attribute, sim_attribute]]
    records = df.to_records(index=False)
    pairs = list(combinations(records, 2))
    return pairs


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



