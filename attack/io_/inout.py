import os
import pickle
from pathlib import Path

import pandas as pd

from classes.graphfile_settings import graphfile_settings


def load_graph_tp(graph_path):
    with open(graph_path, 'rb') as input:
        tpl = pickle.load(input)
    return tpl[0], tpl[1]

def save_graph_tp(graph, true_matches, settings):
    subpath = settings.graph_path
    os.makedirs(subpath, exist_ok=True)
    filename = subpath + "/" + get_filename_template(settings) + ".pkl"
    settings.pickle_file = filename
    tpl = (graph, true_matches)
    delete_graphwave_graph(settings)
    with open(filename, 'wb') as output:
        pickle.dump(tpl, output, pickle.HIGHEST_PROTOCOL)

def get_filename_template(settings):
    if settings.mode != "graph_load":
        nc_suffix = '_nodeCount' if settings.node_count else ''
        comp_suffix = "_comp{}".format(settings.min_comp_size) if not settings.min_comp_size == 3 else ''
        scaling_suffix = ""
        nf_suffix = "_{}".format(settings.node_features)
        lsh_size_suffix = "_lshs{}".format(settings.lsh_size_blk) if settings.lsh_size_blk != 0 else ''
        lsh_count_suffix = "_lshc{}".format(settings.lsh_count_blk) if settings.lsh_size_blk != 0 else ''
        if settings.graph_scaled == 'minmaxscaler':
            scaling_suffix = '_minmax'
        elif settings.graph_scaled == 'standardscaler':
            scaling_suffix = "_std"
        filename = "c{0}_t{1}{9}_r{2}e{3}{4}{5}{6}{7}{8}".format(settings.record_count, settings.threshold,
                                                                  settings.removed_plain_record_frac,
                                                                  settings.removed_encoded_record_frac,
                                                                  lsh_count_suffix,
                                                                  lsh_size_suffix, nf_suffix, nc_suffix, scaling_suffix, comp_suffix)
    else:
        filename = settings.pickle_file.split("/")[-1].split(".pkl")[0]
    return filename

def output_results_csv_features(prec, settings, no_matches):
    common_attr, common_attr_names, df, full_path = output_results_csv_common(no_matches, prec, settings, "features_results.csv")
    # cols: File, Dataset, Record_Count, Threshold, Removed_Records, Node_Features (Mode), Top 10,50,100,500,1000, LSH_Parameter (Count, Size, Cos_Thold, Hyperplane_Count)
    df_append = pd.DataFrame([tuple([*common_attr])], columns=[*common_attr_names])
    df = pd.concat([df, df_append])
    df.to_csv(full_path, index=False)

def output_performance_node_features_results(time, feature_set, settings):
    subpath = settings.graph_path + "/performance"
    os.makedirs(subpath, exist_ok=True)
    filename = subpath + "/nf.csv"
    df = pd.DataFrame([tuple([settings.record_count, settings.threshold, feature_set, time])], columns=['record_count', 'threshold', 'feature_set', 'time'])
    df.to_csv(filename, mode='a', index=False, header = not os.path.isfile(filename))


def output_results_csv_graphwave(prec, settings, no_matches, graphwave_settings, weights):
    common_attr, common_attr_names, df, full_path = output_results_csv_common(no_matches, prec, settings, "graphwave_results.csv")
    # cols: File, Dataset, Record_Count, Threshold, Removed_Records, Node_Features (Normal/Histo/Fast), Top 10,50,100,500,1000, LSH_Parameter (Count, Size, Cos_Thold, Hyperplane_Count)
    sample_pcts_linspace = "(0,{0},{1})".format(str(graphwave_settings.sample_p_max_val), str(graphwave_settings.no_samples))
    graphwave_attr = (graphwave_settings.technique, weights[-1], graphwave_settings.scales, sample_pcts_linspace, graphwave_settings.order_approx)
    graphwave_attr_names = ["type", "weight", "scales", "sample_points_linspace", "order_approx"]
    df_append = pd.DataFrame([tuple([*common_attr, *graphwave_attr])],
                             columns=[*common_attr_names, *graphwave_attr_names])
    df = pd.concat([df, df_append])
    df.to_csv(full_path, index=False)

def output_results_csv_graphsage(prec, settings, no_matches, graphsage_settings, weights):
    common_attr, common_attr_names, df, full_path = output_results_csv_common(no_matches, prec, settings, "graphsage_results.csv")
    # cols: File, Dataset, Record_Count, Threshold, Removed_Records, Node_Features (Normal/Histo/Fast), Top 10,50,100,500,1000, LSH_Parameter (Count, Size, Cos_Thold, Hyperplane_Count)
    graphsage_attr = (weights[-1], graphsage_settings.layers, graphsage_settings.num_samples,
                      graphsage_settings.number_of_walks, graphsage_settings.length,
                      graphsage_settings.batch_size, graphsage_settings.epochs)
    graphsage_attr_names = ["weight", "layers", "num_samples", "number_of_walks", "length",
                            "batch_size", "epochs"]
    df_append = pd.DataFrame([tuple([*common_attr, *graphsage_attr])],
                             columns=[*common_attr_names, *graphsage_attr_names])
    df = pd.concat([df, df_append])
    df.to_csv(full_path, index=False)

def output_results_csv_deepgraphinfomax(prec, settings, no_matches, deepgraphinfomax_settings, weights):
    common_attr, common_attr_names, df, full_path = output_results_csv_common(no_matches, prec, settings, "deepgraphinfomax_results.csv")
    # cols: File, Dataset, Record_Count, Threshold, Removed_Records, Node_Features (Normal/Histo/Fast), Top 10,50,100,500,1000, LSH_Parameter (Count, Size, Cos_Thold, Hyperplane_Count)
    deepgraphinfomax_attr = (weights[-1], deepgraphinfomax_settings.layers, deepgraphinfomax_settings.activations, deepgraphinfomax_settings.epochs)
    deepgraphinfomax_attr_names = ["weight", "layers", "activation", "epochs"]
    df_append = pd.DataFrame([tuple([*common_attr, *deepgraphinfomax_attr])],
                             columns=[*common_attr_names, *deepgraphinfomax_attr_names])
    df = pd.concat([df, df_append])
    df.to_csv(full_path, index=False)


def output_results_csv_common(no_matches, prec, settings, filename):
    os.makedirs(settings.results_path, exist_ok=True)
    output_path = settings.results_path + "/" if settings.results_path[-1] != "/" else settings.results_path
    full_path = output_path + filename
    graph_path = settings.pickle_file.replace("\\", "/")
    graphfile_attr = graphfile_settings(graph_path.split('/')[-1])
    min_comp_size = max(graphfile_attr.min_comp_size, settings.min_comp_size)
    try:
        df = pd.read_csv(full_path, na_filter=False)
        df.rename(columns={"lsh_count": "lsh_count_nm", "lsh_size": "lsh_size_nm"}, inplace=True) #just in case for older files
    except FileNotFoundError:
        df = pd.DataFrame()
    common_attr = (
    settings.pickle_file, graph_path.split('/')[-2], graphfile_attr.record_count, graphfile_attr.threshold,
    graphfile_attr.removed_records_plain, graphfile_attr.removed_records_encoded, graphfile_attr.mode, *prec, no_matches, settings.node_matching_tech,
    settings.lsh_count_nm, settings.lsh_size_nm, settings.cos_sim_thold, settings.hyperplane_count, min_comp_size, settings.scaler, str(settings.vidanage_weights),
    graphfile_attr.lsh_count_blk, graphfile_attr.lsh_size_blk)
    num_top_pairs_str = list(map(lambda i: str(i), settings.num_top_pairs))
    common_attr_names = ['file', 'dataset', 'record_count', 'threshold', 'removed_records_plain', 'removed_records_encoded', 'node_features',
                         *num_top_pairs_str, 'no_matches', 'graph_matching_tech', 'lsh_count_nm', 'lsh_size_nm',
                         'cos_sim_threshold', 'hyperplane_count', 'min_comp_size', 'scaler', 'vidanage_weights', 'lsh_count_blk', 'lsh_size_blk']
    return common_attr, common_attr_names, df, full_path


def get_path_for_graphwave_graph(settings):
    subpath, file = settings.pickle_file.split('/', 1)
    subpath = subpath + "_graphwave"
    return subpath + "/" + file

def graphwave_graph_exists(settings):
    return os.path.exists(get_path_for_graphwave_graph(settings))

def save_graph_for_graphwave_org(graph, settings):
    filename = get_path_for_graphwave_graph(settings)
    subpath, _ = filename.split('/', 1)
    os.makedirs(subpath, exist_ok=True)
    with open(filename, 'wb') as output:
        pickle.dump(graph, output, pickle.HIGHEST_PROTOCOL)

def load_graph_for_graphwave_org(settings):
    filename = get_path_for_graphwave_graph(settings)
    with open(filename, 'rb') as input:
        graph = pickle.load(input)
    return graph

def delete_graphwave_graph(settings):
    if graphwave_graph_exists(settings):
        filename = get_path_for_graphwave_graph(settings)
        os.remove(filename)


