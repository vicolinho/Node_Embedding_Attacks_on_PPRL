import networkx as nx
import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from queue import PriorityQueue

from stellargraph.globalvar import SOURCE, TARGET, WEIGHT

from attack import blocking, embeddings

U = 'u'
V = 'v'

def bipartite_graph_edges_to_matches(G_edges, nodes1, nodes2, no_top_pairs, mode):
    if mode == 'shm':
        return bipartite_graph_edges_to_matches_shm(G_edges, nodes1, nodes2, no_top_pairs)
    elif mode == 'mwm':
        return bipartite_graph_edges_to_matches_mwm(G_edges, nodes1, nodes2, no_top_pairs)
    elif mode == 'smm':
        return bipartite_graph_edges_to_matches_smm(G_edges, nodes1, nodes2, no_top_pairs)


def bipartite_graph_edges_to_matches_mwm(G_edges, nodes1, nodes2, no_top_pairs):
    G = nx.from_pandas_edgelist(G_edges, edge_attr=True)
    highest_pairs = PriorityQueue()
    u = [n for n in G.nodes if n[0] == U]
    try:
        matches = nx.bipartite.minimum_weight_full_matching(G, u)
    except ValueError:
        return []
    for node1, node2 in matches.items():
        if node1[0] == U:
            sim = G.get_edge_data(node1, node2)[WEIGHT]
            if highest_pairs.qsize() == no_top_pairs + 1:
                highest_pairs.get()
            nodeid1 = nodes1[int(node1[1:])]
            nodeid2 = nodes2[int(node2[1:])]
            highest_pairs.put((sim, (nodeid1, nodeid2))) # sim stays reverse so priority_queue return actual highest sim first
    highest_pairs.get() # not correct if no_top_pairs >= pairs
    matches = matches_from_priority_queue(highest_pairs)
    return matches

def embeddings_to_binary_lsh(embeddings1, embeddings2, dim):
    binary_lsh1 = np.zeros((len(embeddings1), dim)).astype(int)
    binary_lsh2 = np.zeros((len(embeddings2), dim)).astype(int)
    C = np.random.randn(len(embeddings1[0]), dim)
    for i in range(0, len(embeddings1)):
        binary_lsh1[i] = embeddings1[i].dot(C) >= 0
    for i in range(0, len(embeddings2)):
        binary_lsh2[i] = embeddings2[i].dot(C) >= 0
    return binary_lsh1, binary_lsh2

def create_dicts_lsh_embeddings(binary_lsh1, binary_lsh2, ids_list_hamming_lsh):
    lsh_dicts_embeddings = []
    for ids_hamming_lsh in ids_list_hamming_lsh:
        lsh_dict = {}
        for i in range(0, len(binary_lsh1)):
            add_id_to_lsh_dict(i, 0, ids_hamming_lsh, lsh_dict, binary_lsh1)
        for i in range(0, len(binary_lsh2)):
            add_id_to_lsh_dict(i, 1, ids_hamming_lsh, lsh_dict, binary_lsh2)
        lsh_dicts_embeddings.append(lsh_dict)
    return lsh_dicts_embeddings


def add_id_to_lsh_dict(i, j, ids_hamming_lsh, lsh_dict, lsh_emb):
    bitvector = blocking.lsh_blocking_key(lsh_emb[i], ids_hamming_lsh)
    #bitvector = [lsh_emb[i][j] for j in ids_hamming_lsh].tobytes()
    if not bitvector in lsh_dict:
        lsh_dict[bitvector] = [OrderedSet(), OrderedSet()]
    lsh_dict[bitvector][j].add(i)

def embeddings_to_bipartite_graph_edges_lsh(embeddings1, embeddings2, settings, weights):
    embeddings1_lsh = embeddings.multiple_embeddings_to_one(embeddings1)
    embeddings2_lsh = embeddings.multiple_embeddings_to_one(embeddings2)
    binary_lsh1, binary_lsh2 = embeddings_to_binary_lsh(embeddings1_lsh, embeddings2_lsh, settings.hyperplane_count)
    ids_list_hamming_lsh = blocking.choose_positions(settings.lsh_count, settings.lsh_size, settings.hyperplane_count)
    lsh_dicts_embeddings = create_dicts_lsh_embeddings(binary_lsh1, binary_lsh2, ids_list_hamming_lsh)
    return embeddings_to_bipartite_graph_edges_lsh_dicts(embeddings1, embeddings2, lsh_dicts_embeddings, settings.cos_sim_thold, weights = weights)



def embeddings_to_bipartite_graph_edges_lsh_dicts(embeddings1, embeddings2, lsh_dicts_embeddings, threshold, vidange = True, weights = [1.0]):
    # lsh_dicts_embeddings: list of dicts (key for comparison)
    # [{bitvector: [[u_ids,],[v_ids,]]}, ...]
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
    if vidange:
        edges = transform_edges_df_to_vidange(edges, w_cos = 0.6, w_sim_conf = 0.3, w_degr_conf = 0.1)
    return edges

def weighted_cos_sim(cos_sims_list, weights):
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
    edge_source_min = edges.loc[edges.groupby([SOURCE])[WEIGHT].idxmin()].reset_index(drop=True)
    edge_target_min = edges.loc[edges.groupby([TARGET])[WEIGHT].idxmin()].reset_index(drop=True)
    shm_edges = pd.merge(edge_source_min, edge_target_min, how='inner', on=edge_source_min.columns.values.tolist())
    shm_edges = shm_edges.sort_values(by=[WEIGHT]).head(no_top_pairs)
    id_mapping_func = lambda x, node_ids: node_ids[int(x[1:])]
    shm_edges[SOURCE] = shm_edges[SOURCE].apply(id_mapping_func, args=([nodes1]))
    shm_edges[TARGET] = shm_edges[TARGET].apply(id_mapping_func, args=([nodes2]))
    matches = list(shm_edges[[SOURCE, TARGET]].to_records(index=False))
    return matches

def bipartite_graph_edges_to_matches_smm(edges, nodes1, nodes2, no_top_pairs):
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

def embeddings_to_bipartite_graph(embeddings1, embeddings2, threshold, func=cosine_similarity): #todo: to be cleaned
    source = []
    target = []
    weight = []
    cos_sims = func(embeddings1, embeddings2)
    for x in range(0, len(embeddings1)):
        for y in range(0, len(embeddings2)):
            sim = cos_sims[x,y]
            if sim >= threshold:
                source.append(U+str(x))
                target.append(V+str(y))
                weight.append(-sim) # needed to use minimum weight matching
    edges = DataFrame({SOURCE: source, TARGET: target, WEIGHT: weight})
    return nx.from_pandas_edgelist(edges, edge_attr=True)

def cos_sim_matrix_to_edges_vidange(cos_sims, threshold, w_cos, w_sim_conf, w_degr_conf):
    w_cos, w_sim_conf, w_degr_conf = normalize_weights_vidange(w_cos, w_degr_conf, w_sim_conf)
    source = []
    target = []
    weight = []
    for x in range(0, len(cos_sims)):
        for y in range(0, len(cos_sims[0])):
            sim = cos_sims[x,y]
            if sim >= threshold:
                source.append(U+str(x))
                target.append(V+str(y))
                weight.append(-sim)
    df = DataFrame({SOURCE: source, TARGET: target, WEIGHT: weight})
    df = transform_edges_df_to_vidange(df, w_cos, w_degr_conf, w_sim_conf)
    return df


def transform_edges_df_to_vidange(df, w_cos, w_degr_conf, w_sim_conf):
    df['cos_sim'] = -1 * df[WEIGHT]
    df['sum_x'] = df['cos_sim'].groupby(df[SOURCE]).transform('sum')
    df['sum_y'] = df['cos_sim'].groupby(df[TARGET]).transform('sum')
    df['count_x'] = df['cos_sim'].groupby(df[SOURCE]).transform('count')
    df['count_y'] = df['cos_sim'].groupby(df[TARGET]).transform('count')
    df['sim_conf'] = (df['cos_sim'] * (df['count_x'] + df['count_y'] - 2) / (
                df['sum_x'] + df['sum_y'] - 2 * df['cos_sim'])).fillna(1.5)
    df['degree_conf'] = 1 / (df['count_x'] + df['count_y'] - 1)
    df = df.drop(columns=['sum_x', 'sum_y', 'count_x', 'count_y'])
    df['cos_sim'] = normalize_col(df['cos_sim'])
    df['sim_conf'] = normalize_col(df['sim_conf'])
    df['degree_conf'] = normalize_col(df['degree_conf'])
    df[WEIGHT] = -1 * (w_cos * df['cos_sim'] + w_sim_conf * df['sim_conf'] + w_degr_conf * df[
        'degree_conf'])  # because of matching algo
    return df


def normalize_col(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df


def embeddings_to_bipartite_graph_vidange(embeddings1, embeddings2, threshold, func=cosine_similarity): #todo: to be cleaned
    cos_sims = func(embeddings1, embeddings2)
    edges = cos_sim_matrix_to_edges_vidange(cos_sims, threshold, w_cos = 0.5, w_sim_conf = 0.3, w_degr_conf = 0.2)
    return nx.from_pandas_edgelist(edges, edge_attr=True)

def normalize_weights_vidange(w_cos, w_degr_conf, w_sim_conf):
    sum_weights = w_cos + w_sim_conf + w_degr_conf
    if sum_weights != 1:
        w_cos /= sum_weights
        w_sim_conf /= sum_weights
        w_degr_conf /= sum_weights
    return w_cos, w_sim_conf, w_degr_conf


def matches_from_embeddings_two_graphs(embeddings1, embeddings2, nodes1, nodes2, settings, weights, prefix_char=False):
    matches = bipartite_graph_edges_to_matches(embeddings_to_bipartite_graph_edges_lsh(embeddings1, embeddings2, settings, weights), nodes1, nodes2, max(settings.num_top_pairs), settings.graph_matching_tech)
    if prefix_char:
        return remove_prefix_from_matches(matches)
    else:
        return matches

def matches_from_embeddings_combined_graph(embedding_results, id1, id2, settings, weights = [1.0]):
    embeddings1, embeddings2, nodes1, nodes2 = split_embeddings_by_nodes(embedding_results.embeddings, embedding_results.nodes, id1, id2)
    return matches_from_embeddings_two_graphs(embeddings1, embeddings2, nodes1, nodes2, settings, weights, True)


def split_embeddings_by_nodes(embeddings, nodes, prefix1, prefix2):
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
    adapted_matches = []
    for match in matches:
        adapted_match = (match[0][2:], match[1][2:])
        adapted_matches.append(adapted_match)
    return adapted_matches

def get_pqueue_pairs_ids_highest_sims(node_embeddings, ids1, ids2, no_top_pairs): # only needed graph1 <-> graph2
    highest_pairs = PriorityQueue()
    for i in ids1:
        for j in ids2:
            cos_sim = cosine_similarity(node_embeddings[i].reshape(1,-1), node_embeddings[j].reshape(1,-1))[0][0]
            if highest_pairs.qsize() == no_top_pairs + 1:
                highest_pairs.get()
            highest_pairs.put((cos_sim, (i, j)))
    highest_pairs.get()
    return highest_pairs

def get_pqueue_pairs_ids_highest_sims_two_graphs(node_embeddings_1, node_embeddings_2, no_top_pairs): # only needed graph1 <-> graph2
    highest_pairs = PriorityQueue()
    for i in range(0, len(node_embeddings_1)):
        for j in range(0, len(node_embeddings_2)):
            cos_sim = cosine_similarity(node_embeddings_1[i].reshape(1,-1), node_embeddings_2[j].reshape(1,-1))[0][0]
            if highest_pairs.qsize() == no_top_pairs + 1:
                highest_pairs.get()
            highest_pairs.put((cos_sim, (i, j)))
    highest_pairs.get()
    return highest_pairs

def matches_from_priority_queue(highest_pairs):
    matches = []
    while highest_pairs.qsize() != 0:
        pair = highest_pairs.get()
        matches.append((pair[1][0],pair[1][1]))
    return matches

def get_matching_node_ids_two_graphs(highest_pairs, node_ids_1, node_ids_2):
    matches = []
    while highest_pairs.qsize() != 0:
        pair = highest_pairs.get()
        matches.append((node_ids_1[pair[1][0]],node_ids_2[pair[1][1]]))
    return matches

def get_pairs_highest_sims_two_graphs(node_embeddings_1, node_embeddings_2, node_ids_1, node_ids_2, no_top_pairs):
    highest_pairs = get_pqueue_pairs_ids_highest_sims_two_graphs(node_embeddings_1, node_embeddings_2, no_top_pairs)
    return get_matching_node_ids_two_graphs(highest_pairs, node_ids_1, node_ids_2)


def get_matching_node_ids(highest_pairs, node_ids):
    while highest_pairs.qsize() != 0:
        pair = highest_pairs.get()
        print(node_ids[pair[1][0]],node_ids[pair[1][1]])


def get_pairs_highest_sims(node_embeddings, node_ids, no_top_pairs):
    ids1, ids2 = split_node_ids_into_two_groups(node_ids, '_')
    get_matching_node_ids(get_pqueue_pairs_ids_highest_sims(node_embeddings, ids1, ids2, no_top_pairs), node_ids)

def split_node_ids_into_two_groups(node_ids, split_char):
    ids1 = []
    ids2 = []
    for pos in range(0,len(node_ids)):
        if str(node_ids[pos])[-1] == split_char:
            ids2.append(pos)
        else:
            ids1.append(pos)
    return ids1, ids2
