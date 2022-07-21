import argparse


def argparser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_full = subparsers.add_parser('graph_calc', help="save graph and calculate precision")
    parser_full.add_argument("--analysis", help='needed if analysis should be conducted', action='store_true')
    parser_full.add_argument("plain_file", help='path to plain dataset')
    parser_full.add_argument("encoded_file", help='path to encoded dataset')
    parser_full.add_argument("results_path", help='path to results output file')
    parser_full.add_argument("threshold", help='similarity threshold to be included in graph')
    parser_full.add_argument("--graph_path", help='path where graph is to be stored', default='graphs')
    parser_full.add_argument("--remove_frac_plain", help='fraction of plain records to be excluded')
    parser_full.add_argument("--record_count", help='restrict record count to be processed')
    parser_full.add_argument("--histo_features", help='adds histograms as features', action='store_true')
    parser_full.add_argument("--fast_mode", help='fast mode without time-consuming node features', action='store_true')
    parser_full.add_argument("--lsh_size", help='vector size for hamming lsh for indexing', default=0)
    parser_full.add_argument("--lsh_count", help='count of different lsh vectors for indexing', default=1)
    parser_full.add_argument("--min_edges", help='minimum edge count for a node to be matched', default=0)
    parser_full.add_argument("--graph_matching_tech", help='graph matching technique (shm, mwm, smm)', default='shm')


    parser_save_graph = subparsers.add_parser("graph_load", help='loading instead of calculating sim graph')
    parser_save_graph.add_argument("pickle_file", help="path to pickle file with graph and true matches")
    parser_save_graph.add_argument("results_path", help='path to results output file')
    parser_save_graph.add_argument("--lsh_size", help='vector size for hamming lsh for indexing', default=0)
    parser_save_graph.add_argument("--lsh_count", help='count of different lsh vectors for indexing', default=1)
    parser_save_graph.add_argument("--graph_matching_tech", help='graph matching technique (shm, mwm, smm)', default='shm')
    parser_save_graph.add_argument("--min_edges", help='minimum edge count for a node to be matched', default=0)
    parser_save_graph.add_argument("--graphwave_libpath", help='path to original graphwave libpath')
    #parser_save_graph.add_argument("--graphsage_settings_file", help='path to graphsage settings file for hyperparameter tuning')
    #parser_save_graph.add_argument("--deepgraphinfomax_settings_file", help='path to deepgraphinfomax settings file for hyperparameter tuning')
    args = parser.parse_args()
    return args