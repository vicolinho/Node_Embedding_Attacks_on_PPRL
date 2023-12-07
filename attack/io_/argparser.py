import argparse

def argparser():
    """
    creates argument parser and returns arguments
    :return: argparser.Namespace: stores arguments from parser
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')
    parser_calc = subparsers.add_parser('graph_save', help="create and save similarity graph with node features")
    parser_load = subparsers.add_parser("graph_load", help='analysis with loading instead of calculating sim graph')
    parser_both = subparsers.add_parser("graph_save_calculate", help="all-in-one: save graph and calculate precision")

    for subparser in [parser_calc, parser_both]:
        subparser.add_argument("plain_file", help='path to plain dataset')
        subparser.add_argument("encoded_file", help='path to encoded dataset')
        subparser.add_argument("threshold", help='similarity threshold to be included in graph', type=fraction)
        subparser.add_argument("--graph_path", help='path where graph is to be stored', default='graphs')
        subparser.add_argument("--remove_frac_plain", help='fraction of plain records to be excluded', default=0.0, type=fraction)
        subparser.add_argument("--remove_frac_encoded", help='fraction of encoded records to be excluded',
                                 default=0.0, type=fraction)
        subparser.add_argument("--record_count", help='restrict record count to be processed, if negative will look from bottom up')
        subparser.add_argument("--node_features",
                                 help='which Vidanage-based node features will be used (all, fast, egonet1, egonet2)',
                                 default="all", choices=['all', 'fast', 'egonet1', 'egonet2'])
        subparser.add_argument("--node_count", help='includes node count as node feature', action='store_true', default=False)
        subparser.add_argument("--nf_scaled",
                               help='needs to be set if both graphs should be scaled independently (standardscaler, minmaxscaler)',
                               default="", choices=['standardscaler', 'minmaxscaler'])
        subparser.add_argument("--min_comp_size", help='minimum connected component size for node to be matched',
                               default=3, type=int)
        subparser.add_argument("--padding", help="if set creates bigrams with padding, must be compatible to encoded data", action='store_true', default=False)
        subparser.add_argument("--lsh_size_blocking", help='vector size for hamming lsh for indexing (blocking)', default=0, type=int)
        subparser.add_argument("--lsh_count_blocking", help='count of different lsh vectors for indexing (blocking)', default=1, type=int)
        subparser.add_argument("--qgram_attributes", help="names of the attributes of which qgrams should be created (plain file)", nargs='*', default=['first_name', 'last_name'])
        subparser.add_argument("--encoded_attr", help="name of the attribute with the encoded (base64) bloom filter (encoded file)", nargs='*', default='base64_bf')

    parser_load.add_argument("pickle_file", help="path to pickle file with graph and true matches")
    parser_load.add_argument("--min_comp_size", help='minimum connected component size for node to be matched', default=0, type=int)

    for subparser in [parser_load, parser_both]:
        subparser.add_argument("--results_path", help='path to results output file', default="results")
        subparser.add_argument("--lsh_size_node_matching", help='vector size for hamming lsh for indexing (node matching)', default=0, type=int)
        subparser.add_argument("--lsh_count_node_matching", help='count of different lsh vectors for indexing (node matching)', default=1, type=int)
        subparser.add_argument("--node_matching_tech", help='node matching technique (shm, mwm, smm)',
                                 default='shm')
        default_weights = [i / 10 for i in range(1, 10)]
        subparser.add_argument("--weight_list", help='list of weights to combine node features with node embeddings',
                                 nargs='*', type=fraction, default=default_weights)
        subparser.add_argument("--graphwave_sg_lib", help='to use StellarGraph Implementation for GraphWave (w/o edge weights)', action="store_true")
        subparser.add_argument("--hp_config_file",
                                 help='config file in config directory to import hyperparameters (without .py)',
                                 default='config')
        subparser.add_argument("--scaler", help='scaling mode for node embeddings (standardscaler, minmaxscaler)',
                                 default='standardscaler', choices=['standardscaler', 'minmaxscaler'])
        subparser.add_argument("--num_top_pairs", help='list with numbers of top matching pairs to be evaluated', nargs='*', type=int, default=[10, 50, 100, 500, 1000])
        subparser.add_argument("--node_matching_threshold",
                                 help='threshold for cosinus sim in the node matching step',
                                 default=0.9, type=float)
        subparser.add_argument("--vidanage_weights",
                                 help='weights for similarity calculation (cosinus sim, sim conf, degree conf)',
                                 nargs=3,
                                 type=float, default=[0.6, 0.3, 0.1])

    args = parser.parse_args()
    return args

def fraction(x):
    """
    type for parser: float between 0.0 and 1.0
    """
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x > 1.0 or x < 0.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" %(x,))
    return x