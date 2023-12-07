import networkx as nx
import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from queue import PriorityQueue

from stellargraph.globalvar import SOURCE, TARGET, WEIGHT

from attack.blocking import blocking
from attack.node_embeddings import embeddings

U = 'u'
V = 'v'

def bipartite_graph_edges_to_matches(G_edges, nodes1, nodes2, no_top_pairs, mode):
    """
    calculates matches with bipartite graph data
    :param G_edges (pd.DataFrame): edge data for bipartite graph
    :param nodes1 (list of str): list of encoded nodes (with prefix)
    :param nodes2 (list of str): list of encoded nodes (with prefix)
    :param no_top_pairs (int): number of matches to be selected
    :param mode (str): abbreviation for node matching mode (shm - symmetric highest match, mwm - maximum weight match, smm - stable marriage match)
    :return: list of records: matches with node ids (still with prefix)
    """
    if mode == 'shm':
        return bipartite_graph_edges_to_matches_shm(G_edges, nodes1, nodes2, no_top_pairs)
    elif mode == 'mwm':
        return bipartite_graph_edges_to_matches_mwm(G_edges, nodes1, nodes2, no_top_pairs)
    elif mode == 'smm':
        return bipartite_graph_edges_to_matches_smm(G_edges, nodes1, nodes2, no_top_pairs)


def bipartite_graph_edges_to_matches_mwm(G_edges, nodes1, nodes2, no_top_pairs):
    """
    selects matches with maximum weight matching (uses networkx minimum weight matching, that's why edge weights were inverted)
    :param edges (pd.DataFrame): edge data of bipartite graph (with weights)
    :param nodes1 (list of str): ids of plain nodes
    :param nodes2 (list of str): ids of encoded nodes
    :param no_top_pairs (int): number of matches to be selected
    :return: list of records: matches with node ids
    """
    G = nx.from_pandas_edgelist(G_edges, edge_attr=True)
    highest_pairs = PriorityQueue()
    u = [n for n in G.nodes if n[0] == U]
    try:
        matches = nx.bipartite.minimum_weight_full_matching(G, u)
    except ValueError: # if cost matrix isn't feasible
        return []
    for node1, node2 in matches.items():
        if node1[0] == U:
            sim = G.get_edge_data(node1, node2)[WEIGHT]
            nodeid1 = nodes1[int(node1[1:])]
            nodeid2 = nodes2[int(node2[1:])]
            highest_pairs.put((-sim, (nodeid1, nodeid2))) # sim back to normal one (because of MINIMUM weight matching)
            if highest_pairs.qsize() == no_top_pairs + 1:
                highest_pairs.get()
    matches = matches_from_priority_queue(highest_pairs)
    return matches

def embeddings_to_binary_lsh(embeddings1, embeddings2, dim):
    """
    calculates binary lsh vectors of embeddings
    :param embeddings1 (list of ndarray): node embedding and/or node features of plain data
    :param embeddings2 (list of ndarray): node embedding and/or node features of encoded data
    :param dim (int): length of lsh vector
    :return: 2d-ndarray (lsh vectors of plain nodes), 2d-ndarray (lsh vectors of encoded nodes)
    """
    binary_lsh1 = np.zeros((len(embeddings1), dim)).astype(int)
    binary_lsh2 = np.zeros((len(embeddings2), dim)).astype(int)
    C = np.random.randn(len(embeddings1[0]), dim)
    for i in range(0, len(embeddings1)):
        binary_lsh1[i] = embeddings1[i].dot(C) >= 0
    for i in range(0, len(embeddings2)):
        binary_lsh2[i] = embeddings2[i].dot(C) >= 0
    return binary_lsh1, binary_lsh2

def create_dicts_lsh_embeddings(binary_lsh1, binary_lsh2, ids_list_hamming_lsh):
    """
    stores different dicts for different lsh runs, dict consists of lsh key and two sets of node ids for plain and encoded nodes
    :param binary_lsh1 (ndarray, dim: 2): lsh vectors of plain nodes
    :param binary_lsh2 (ndarray, dim: 2): lsh vectors of encoded nodes
    :param ids_list_hamming_lsh (list of list of int): selected positions of binary vectors
    :return: list of dict: stores different dicts for different lsh runs, dict consists of lsh key and two sets of node ids for plain and encoded nodes
    """
    lsh_dicts_embeddings = []
    for ids_hamming_lsh in ids_list_hamming_lsh:
        lsh_dict = {}
        for i in range(0, len(binary_lsh1)):
            add_id_to_lsh_dict(i, False, ids_hamming_lsh, lsh_dict, binary_lsh1)
        for i in range(0, len(binary_lsh2)):
            add_id_to_lsh_dict(i, True, ids_hamming_lsh, lsh_dict, binary_lsh2)
        lsh_dicts_embeddings.append(lsh_dict)
    return lsh_dicts_embeddings


def add_id_to_lsh_dict(i, encoded, ids_hamming_lsh, lsh_dict, lsh_emb):
    """
    adds node id to fitting set of node ids (by plain/encoded and lsh key)
    :param i (int): index number of node
    :param encoded (bool): if set of encoded nodes is used or not/plain
    :param ids_hamming_lsh (list of int): ids of bits which will be used for LSH key
    :param lsh_dict (dict: (key: bytes, value: [OrderedSet(int), OrderedSet(int)]): stores set of node ids for plain and encoded nodes seperated and by bitstring
    :param lsh_emb (ndarray, dim: 2): array of embeddings transformed to bitarrays for lsh
    """
    j = 1 if encoded else 0
    bitvector = blocking.lsh_blocking_key(lsh_emb[i], ids_hamming_lsh)
    if not bitvector in lsh_dict:
        lsh_dict[bitvector] = [OrderedSet(), OrderedSet()]
    lsh_dict[bitvector][j].add(i)

def embeddings_to_bipartite_graph_edges_lsh(embeddings1, embeddings2, settings, weights):
    """
    takes node embeddings and returns dataframe with edge data for bipartite graph used for node matching
    :param embeddings1 (list of ndarray): node embedding and/or node features of plain data
    :param embeddings2 (list of ndarray): node embedding and/or node features of encoded data
    :param settings (Settings)
    :param weights (list of float): to get weighted similarity of node embeddings/features
    :return: pd.DataFrame (with source, target and weight of nodes)
    """
    embeddings1_lsh = embeddings.multiple_embeddings_to_one(embeddings1)
    embeddings2_lsh = embeddings.multiple_embeddings_to_one(embeddings2)
    binary_lsh1, binary_lsh2 = embeddings_to_binary_lsh(embeddings1_lsh, embeddings2_lsh, settings.hyperplane_count)
    ids_list_hamming_lsh = blocking.choose_positions(settings.lsh_count_nm, settings.lsh_size_nm, settings.hyperplane_count)
    lsh_dicts_embeddings = create_dicts_lsh_embeddings(binary_lsh1, binary_lsh2, ids_list_hamming_lsh)
    return embeddings_to_bipartite_graph_edges_lsh_dicts(embeddings1, embeddings2, lsh_dicts_embeddings, settings.cos_sim_thold, settings.vidanage_weights, weights)



def embeddings_to_bipartite_graph_edges_lsh_dicts(embeddings1, embeddings2, lsh_dicts_embeddings, threshold, vidanage_weights, weights):
    """
    takes node embeddings and lsh division for comparisons and returns dataframe with edge data for bipartite graph used for node matching
    :param embeddings1 (list of array): embeddings for plain nodes
    :param embeddings2 (list of array): embeddings for encoded nodes
    :param lsh_dicts_embeddings (list of dict(key: bytes, value: [OrderedSet(int), OrderedSet(int)]: structure to select index of nodes whose node embeddings are to be compared
    :param threshold (float): threshold for cosine similarity
    :param vidanage_weights ([float, float, float]): weights for different similarity measures (cosine similarity, similarity and degree confidence)
    :param weights (list of float): weights for weighted similarity between node features/embeddings
    :return: pd.DataFrame (with source, target and weight of nodes)
    """
    source, target, weight = [], [], []
    for lsh_dict_embeddings in lsh_dicts_embeddings:
        for value in lsh_dict_embeddings.values():
            cos_sims_list = []
            for j in range(0, len(embeddings1)):
                temp_embeddings1 = [embeddings1[j][i] for i in value[0]]
                temp_embeddings2 = [embeddings2[j][i] for i in value[1]]
                if len(temp_embeddings1) == 0 or len(temp_embeddings2) == 0:
                    continue
                cos_sims = cosine_similarity(temp_embeddings1, temp_embeddings2)
                cos_sims_list.append(cos_sims)
            cos_sims = weighted_cos_sim(cos_sims_list, weights)
            for x in range(0, len(cos_sims)):
                for y in range(0, len(cos_sims[x])):
                    sim = cos_sims[x, y]
                    if sim >= threshold:
                        source.append(U + str(value[0][x]))
                        target.append(V + str(value[1][y]))
                        weight.append(-sim)  # needed to use minimum weight matching
    edges = DataFrame({SOURCE: source, TARGET: target, WEIGHT: weight})
    if weights != [1.0, 0.0, 0.0]:
        edges = transform_edges_df_to_vidanage(edges, w_cos = vidanage_weights[0], w_sim_conf = vidanage_weights[1], w_degr_conf = vidanage_weights[2])
    return edges

def weighted_cos_sim(cos_sims_list, weights):
    """
    merge cosine similarity matrices to one (with weights)
    :param cos_sims_list (list of 2d array): matrix of pairwise cosine similarities, two matrices if using combination of node features and node embeddings
    :param weights (list of float): weights for node features and node embeddings for weighted cosine similarity
    :return: 2d array: matrix of pairwise weighted cosine similarities
    """
    try:
        if len(cos_sims_list) == 0:
            return []
        else:
            cos_sim = weights[0] * cos_sims_list[0]
    except IndexError:
        print("IndexError", cos_sims_list)
        return []
    for i in range(1, len(cos_sims_list)):
        cos_sim += weights[i] * cos_sims_list[i]
    return cos_sim

def bipartite_graph_edges_to_matches_shm(edges, nodes1, nodes2, no_top_pairs):
    """
    selects matches with symmetric highest matching
    :param edges (pd.DataFrame): edge data of bipartite graph (with weights)
    :param nodes1 (list of str): ids of plain nodes
    :param nodes2 (list of str): ids of encoded nodes
    :param no_top_pairs (int): number of matches to be selected
    :return: list of records: matches with node ids
    """
    if len(edges) == 0:
        return []
    try:
        edge_source_min = edges.loc[edges.groupby([SOURCE])[WEIGHT].idxmin()].reset_index(drop=True)
        edge_target_min = edges.loc[edges.groupby([TARGET])[WEIGHT].idxmin()].reset_index(drop=True)
    except KeyError:
        return []
    shm_edges = pd.merge(edge_source_min, edge_target_min, how='inner', on=edge_source_min.columns.values.tolist())
    shm_edges = shm_edges.sort_values(by=[WEIGHT]).head(no_top_pairs)
    id_mapping_func = lambda x, node_ids: node_ids[int(x[1:])]
    shm_edges[SOURCE] = shm_edges[SOURCE].apply(id_mapping_func, args=([nodes1]))
    shm_edges[TARGET] = shm_edges[TARGET].apply(id_mapping_func, args=([nodes2]))
    matches = list(shm_edges[[SOURCE, TARGET]].to_records(index=False))
    return matches

def bipartite_graph_edges_to_matches_smm(edges, nodes1, nodes2, no_top_pairs):
    """
    selects matches with stable marriage matching
    :param edges (pd.DataFrame): edge data of bipartite graph (with weights)
    :param nodes1 (list of str): ids of plain nodes
    :param nodes2 (list of str): ids of encoded nodes
    :param no_top_pairs (int): number of matches to be selected
    :return: list of records: matches with node ids
    """
    # 2 DataFrame u, v (as indices): current_partner, preference list
    PREF_LIST = 'pref_list'
    CURRENT_PARTNER = 'current_partner'
    edges[TARGET + WEIGHT] = list(zip(edges[TARGET], edges[WEIGHT]))
    pref_source = edges.sort_values(by=[SOURCE, WEIGHT]).groupby(SOURCE, sort=False)[TARGET+WEIGHT].agg(list)
    pref_target = edges.sort_values(by=[TARGET, WEIGHT]).groupby(TARGET, sort=False)[SOURCE].agg(list)
    dummy_col = pd.Series(len(pref_source) * [None], index=pref_source.index)
    df_source = pd.concat([dummy_col, dummy_col.copy(), pref_source], axis=1, keys=[CURRENT_PARTNER, WEIGHT, PREF_LIST])
    dummy_col = pd.Series(len(pref_target) * [None], index=pref_target.index)
    df_target = pd.concat([dummy_col, dummy_col.copy(), pref_target], axis=1, keys=[CURRENT_PARTNER, WEIGHT, PREF_LIST])
    left_nodes_with_possible_matches = list(df_source.index)
    while len(left_nodes_with_possible_matches) > 0:
        u = left_nodes_with_possible_matches[0]
        v_tuple = df_source.loc[u, PREF_LIST].pop(0)
        v = v_tuple[0]
        v_partner = df_target.loc[v, CURRENT_PARTNER]
        if v_partner == None:
            df_source.loc[u, CURRENT_PARTNER] = v
            df_source.loc[u, WEIGHT] = v_tuple[1]
            df_target.loc[v, CURRENT_PARTNER] = u
            left_nodes_with_possible_matches.pop(0)
        elif df_target.loc[v, PREF_LIST].index(u) < df_target.loc[v, PREF_LIST].index(v_partner):
            df_source.loc[u, CURRENT_PARTNER] = v
            df_source.loc[u, WEIGHT] = v_tuple[1]
            df_source.loc[v_partner, CURRENT_PARTNER] = None
            df_source.loc[v_partner, WEIGHT] = None
            df_target.loc[v, CURRENT_PARTNER] = u
            left_nodes_with_possible_matches.pop(0)
            if len(df_source.loc[v_partner, PREF_LIST]) != 0:
                left_nodes_with_possible_matches.insert(0, v_partner)
        else:
            if len(df_source.loc[u, PREF_LIST]) == 0:
                left_nodes_with_possible_matches.pop(0)
    id_mapping_func = lambda x, node_ids: node_ids[int(x[1:])]
    df_source = df_source.dropna()
    df_source[SOURCE] = df_source.index.to_series().apply(id_mapping_func, args=([nodes1]))
    df_source[TARGET] = df_source[CURRENT_PARTNER].apply(id_mapping_func, args=([nodes2]))
    df_source = df_source.sort_values(by=WEIGHT)
    matches = list(df_source[[SOURCE, TARGET]].to_records(index=False))
    return matches[:no_top_pairs]


def transform_edges_df_to_vidanage(df, w_cos, w_degr_conf, w_sim_conf):
    """
    calculates edge weights for bipartite graph for node matching
    :param df (pd.Dataframe): stores edge data for bipartite graph
    :param w_cos (float): weight of cosine similarity
    :param w_degr_conf (float): weight of degree confidence
    :param w_sim_conf (float): weight of similarity confidence
    :return: pd.DataFrame
    """
    assert w_cos + w_degr_conf + w_sim_conf == 1.0
    df['cos_sim'] = -1 * df[WEIGHT]
    df['sum_x'] = df['cos_sim'].groupby(df[SOURCE]).transform('sum')
    df['sum_y'] = df['cos_sim'].groupby(df[TARGET]).transform('sum')
    df['count_x'] = df['cos_sim'].groupby(df[SOURCE]).transform('count')
    df['count_y'] = df['cos_sim'].groupby(df[TARGET]).transform('count')
    df['sim_conf'] = (df['cos_sim'] * (df['count_x'] + df['count_y'] - 2) / (
                df['sum_x'] + df['sum_y'] - 2 * df['cos_sim'])).fillna(1.5)
    df['degree_conf'] = 1 / (df['count_x'] + df['count_y'] - 1)
    df.drop(columns=['sum_x', 'sum_y', 'count_x', 'count_y'], inplace = True)
    df['cos_sim'] = normalize_col(df['cos_sim'])
    df['sim_conf'] = normalize_col(df['sim_conf'])
    df['degree_conf'] = normalize_col(df['degree_conf'])
    df[WEIGHT] = -1 * (w_cos * df['cos_sim'] + w_sim_conf * df['sim_conf'] + w_degr_conf * df[
        'degree_conf'])  # because of (minimal weight) matching algo
    return df


def normalize_col(df):
    """
    does minmax normalisation with DataFrame or just a column
    :param df (pd.DataFrame or pd.Series)
    :return: (pd.DataFrame or pd.Series)
    """
    df = (df - df.min()) / (df.max() - df.min())
    return df




def matches_from_embeddings_two_graphs(embeddings1, embeddings2, nodes1, nodes2, settings, weights, prefix_char=False):
    """
    calculates final matches out of node embeddings
    :param embeddings1 (list of ndarray): node embedding and/or node features of plain data
    :param embeddings2 (list of ndarray): node embedding and/or node features of encoded data
    :param nodes1 (list of str): plain node ids
    :param nodes2 (list of str): encoded node ids
    :param settings (Settings)
    :param weights (list of float): to get weighted similarity of node embeddings/features
    :param prefix_char (bool): are prefix characters to be removed
    :return: list of records: matches (without prefix needed previously for distinction)
    """
    matches = bipartite_graph_edges_to_matches(embeddings_to_bipartite_graph_edges_lsh(embeddings1, embeddings2, settings, weights), nodes1, nodes2, max(settings.num_top_pairs), settings.node_matching_tech)
    if prefix_char:
        return remove_prefix_from_matches(matches)
    else:
        return matches

def matches_from_embeddings_combined_graph(embedding_results, id1, id2, settings, weights = [1.0]):
    """
    selects matches from embeddings
    :param embedding_results (Embedding_results)
    :param id1 (str): prefix for plain nodes
    :param id2 (str): prefix for encoded nodes
    :param settings (Settings)
    :param weights (list of float): weights for weighted similarites of node features and node embeddings
    :return: list of tuples: selected matches
    """
    embeddings1, embeddings2, nodes1, nodes2 = split_embeddings_by_nodes(embedding_results.embeddings, embedding_results.nodes, id1, id2)
    return matches_from_embeddings_two_graphs(embeddings1, embeddings2, nodes1, nodes2, settings, weights, True)


def split_embeddings_by_nodes(embeddings, nodes, prefix1, prefix2):
    """
    divides embeddings and nodes depending if encoded or not
    :param embeddings (list of ndarray): stores node embeddings and/or node features
    :param nodes (list of str): ids of nodes
    :param prefix1 (str): prefix of plain nodes
    :param prefix2 (str): prefix of encoded nodes
    :return: list of ndarray (list of plain embeddings), list of ndarray (list of encoded embeddings), list of str (plain node list), list of str (encoded node list)
    """
    embeddings1 = []
    embeddings2 = []
    for i in range(0, len(embeddings)):
        embeddings1.append([])
        embeddings2.append([])
    nodes1 = []
    nodes2 = []
    for i in range(0, len(nodes)):
        if nodes[i][0] == prefix1:
            nodes1.append(nodes[i])
            for j in range(0, len(embeddings)):
                embeddings1[j].append(embeddings[j][i])
        elif nodes[i][0] == prefix2:
            nodes2.append(nodes[i])
            for j in range(0, len(embeddings)):
                embeddings2[j].append(embeddings[j][i])
    return embeddings1, embeddings2, nodes1, nodes2


def remove_prefix_from_matches(matches):
    """
    removes prefix from node ids in matches
    :param matches (list of records)
    :return: list of records: matches without prefix
    """
    adapted_matches = []
    for match in matches:
        adapted_match = (match[0][2:], match[1][2:])
        adapted_matches.append(adapted_match)
    return adapted_matches

def matches_from_priority_queue(highest_pairs):
    """
    returns matches as list from priority queue
    :param highest_pairs (PriorityQueue): matches with weight as priority
    :return: list of records: matches with node ids
    """
    matches = []
    while highest_pairs.qsize() != 0:
        pair = highest_pairs.get()
        matches.append((pair[1][0],pair[1][1]))
    return list(reversed(matches)) #reversed because priority queue return lowest value first, but we want the most probable at first

