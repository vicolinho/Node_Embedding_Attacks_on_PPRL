import pandas as pd
from itertools import combinations

from pandas import DataFrame
from stellargraph.globalvar import SOURCE, TARGET, WEIGHT

from attack.constants import QGRAMS, BITARRAY, LSH_BLOCKING
from attack.sim_graph import sim_graph
from attack.sim_graph.adjust_sims import compute_adjusted_dice_from_bits


def edges_df_from_blk_plain(blk_dict, threshold, id):
    """
    calculates edge data for plain similarity graph
    :param blk_dict (dict: (key: bytes, value: pd.DataFrame)): dict to seperate data where dice similarity is to be compared (blocking)
    :param threshold (float): threshold for similarity graph
    :param id (str): prefix to distinguish plain from encoded nodes
    :return: pd.DataFrame: edge data for similarity graph
    """
    df = DataFrame()
    for key, value in blk_dict.items():
        df_temp = edges_df_from_blk_element(value, threshold, id=id, node_attribute=QGRAMS, sim_attribute=QGRAMS,
                                  sim_func=dice_sim_plain)
        df = pd.concat([df, df_temp])
    return df

def edges_df_from_blk_bf_adjusted(blk_dict, threshold, encoded_attr, bf_length, num_of_hash_func, id):
    """
    calculates edge data for encoded similarity graph
    :param blk_dict (dict: (key: bytes, value: pd.DataFrame)): dict to seperate data where dice similarity is to be compared (blocking)
    :param threshold (float): threshold for similarity graph
    :param encoded_attr (str): attribute name of encoded bloom filter
    :param bf_length (int): length of bloom filter
    :param num_of_hash_func (int): number of hash functions used for bloom filter
    :param id (str): prefix to distinguish plain from encoded nodes
    :return: pd.DataFrame: edge data for similarity graph
    """
    df = DataFrame()
    for key, value in blk_dict.items():
        df_temp = edges_df_from_blk_element(value, threshold, id=id, node_attribute=encoded_attr, sim_attribute=BITARRAY,
                                            sim_func=compute_adjusted_dice_from_bits, num_hash_f=num_of_hash_func,
                                            bf_length=bf_length)

        df = pd.concat([df, df_temp])
    return df


def edges_df_from_blk_element(df, threshold, node_attribute, sim_attribute, sim_func, id, **kwargs):
    """
    calculates edges of similarity graph based of block
    :param df (pd.DataFrame): qgram data
    :param threshold (float): threshold for similarity graph
    :param node_attribute (str): name of attribute used for node representation
    :param sim_attribute (str): name of attribute used for similarity calculation
    :param sim_func (function): similarity function to be used
    :param id (str): prefix to distinguish plain from encoded nodes
    :param kwargs (dict): possibly needed for calculation for encoded data (bloom filter length, hash function count)
    :return: pd.DataFrame with edge data
    """
    arr_first, arr_second, arr_sims = [], [], []
    number = (df.columns[0]).find(LSH_BLOCKING) + len(LSH_BLOCKING)
    i = int(df.columns[0][number:])
    columns = df.columns[-i:] if i > 0 else []
    df = df.loc[:, [node_attribute, sim_attribute, *columns]]
    df.columns.values[0] = 'node_attribute'
    df.columns.values[1] = 'sim_attribute' # can't have same column name twice
    records = df.to_records(index=False)  # needed for using combinations function
    pairs = list(combinations(records, 2))
    for pair in pairs:
        if not pair_computed(pair):
            sim = sim_func(pair[0][1], pair[1][1], **kwargs)
            if sim >= threshold:
                arr_first.append(sim_graph.adjust_node_id(pair[0][0], id))
                arr_second.append(sim_graph.adjust_node_id(pair[1][0], id))
                arr_sims.append(sim)
    d = {SOURCE: arr_first, TARGET: arr_second , WEIGHT: arr_sims}
    return pd.DataFrame(d)


def pair_computed(pair):
    """
    checks if combination pair is already computed
    :param pair (tuple of frozenset or bitarray)
    :return: bool: is pair already compared
    """
    for i in range(2, len(pair[0])):
        if pair[0][i] == pair[1][i]:
            return True
    return False


def dice_sim_plain(bigrams1, bigrams2):
    """
    calculates dice similarity of bigram sets
    :param bigrams1 (set of tuples (which consist of characters)): bigrams to be compared
    :param bigrams2 (set of tuples (which consist of characters))
    :return: float: dice similarity
    """
    intersection = bigrams1.intersection(bigrams2)
    if len(bigrams1) + len(bigrams2) != 0:
        return 2 * len(intersection) / (len(bigrams1) + len(bigrams2))
    else:
        return 0.0



