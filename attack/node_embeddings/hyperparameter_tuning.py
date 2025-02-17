from stellargraph import StellarGraph

from attack.io_ import inout
from attack.node_embeddings import embeddings
from classes.deepgraphinfomax_settings import Deepgraphinfomax_settings
from classes.graphsage_settings import Graphsage_settings
from classes.graphwave_settings import Graphwave_settings

def embeddings_hyperparameter_deepgraphinfomax_gen(G, deepgraphinfomax_settings_list, scaler):
    """

    returns generator with node embeddings for DeepGraphInfomax depending on its hyperparameter settings
    :param G (StellarGraph): similarity graph with node features
    :param deepgraphinfomax_settings_list (list of Deepgraphinfomax_settings): settings for hyperparameter tuning
    :param scaler (StandardScaler or MinMaxScaler): to scale node embeddings
    :return: generator
    """
    for deepgraphinfomax_settings in deepgraphinfomax_settings_list:
        yield embeddings.generate_node_embeddings_deepgraphinfomax(G, deepgraphinfomax_settings, scaler)

def get_params_deepgraphinfomax(config):
    """
    reads DeepGraphInfomax hyperparameters into list of Deepgraphinfomax_settings
    :param config (module): module where hyperparameter settings are set
    :return: list of Deepgraphinfomax_settings: stores hyperparameter for DeepGraphInfomax
    """
    layer_structures_list = config.deepgraphinfomax_settings[config.LAYER_STRUCTURES_LIST]
    activation_func_list = config.deepgraphinfomax_settings[config.ACTIVATION_FUNC_LIST]
    epochs_list = get_default_params_deepgraphinfomax(config)
    deepgraphinfomax_settings_list = []
    for i in range(0, len(layer_structures_list)):
        deepgraphinfomax_settings = Deepgraphinfomax_settings(layers=layer_structures_list[i], activations=activation_func_list[i], epochs=epochs_list[i])
        deepgraphinfomax_settings_list.append(deepgraphinfomax_settings)
    return deepgraphinfomax_settings_list

def get_default_params_deepgraphinfomax(config):
    """
    sets (optional) values for graphsage hyperparameter tuning,
     default: epochs = 20
    :param config (module): module where hyperparameter settings are set
    :return: list of int: list of epoch counts
    """
    configs_count = len(config.deepgraphinfomax_settings[config.LAYER_STRUCTURES_LIST])
    try:
        epochs_list = config.deepgraphinfomax_settings[config.EPOCHS_LIST]
    except Exception:
        epochs_list = configs_count * [20]
    return epochs_list

def embeddings_hyperparameter_graphsage_gen(G, graphsage_settings_list, scaler):
    """
    returns generator with node embeddings for GraphSAGE depending on its hyperparameter settings
    :param G (StellarGraph): similarity graph with node features
    :param graphsage_settings_list (list of Graphsage_settings): settings for hyperparameter tuning
    :param scaler (StandardScale or MinMaxScaler): to scale node embeddings (with sklearn)
    :return: generator
    """
    for graphsage_settings in graphsage_settings_list:
        yield embeddings.generate_node_embeddings_graphsage(G, graphsage_settings, scaler)

def get_params_graphsage(config):
    """
    reads GraphSAGE hyperparameters into list of Graphsage_settings
    :param config (module): module where hyperparameter settings are set
    :return: list of Graphsage_settings: stores hyperparameter for GraphSAGE
    """
    layer_structures_list = config.graphsage_settings[config.LAYER_STRUCTURES_LIST]
    num_samples_list = config.graphsage_settings[config.NUM_SAMPLES_LIST]
    num_walks_list, length_list, batch_size_list, epochs_list = get_default_params_graphsage(config)
    graphsage_settings_list = []
    for i in range(0, len(layer_structures_list)):
        graphsage_settings = Graphsage_settings(layers=layer_structures_list[i], num_samples=num_samples_list[i],
                                                number_of_walks = num_walks_list[i], length = length_list[i],
                                                batch_size = batch_size_list[i], epochs = epochs_list[i])
        graphsage_settings_list.append(graphsage_settings)
    return graphsage_settings_list

def get_default_params_graphsage(config):
    """
    sets (optional) values for graphsage hyperparameter tuning,
     default: number_of_walks = 1, length = 5, batch_size = 512, epochs = 10
    :param config (module): module where hyperparameter settings are set
    :return: num_walks_list, length_list, batch_size_list, epochs_list (all list of int)
    """
    configs_count = len(config.graphsage_settings[config.LAYER_STRUCTURES_LIST])
    try:
        num_walks_list = config.graphsage_settings[config.NUM_WALKS_LIST]
    except Exception:
        num_walks_list = configs_count * [1]
    try:
        length_list = config.graphsage_settings[config.LENGTH_LIST]
    except Exception:
        length_list = configs_count * [5]
    try:
        batch_size_list = config.graphsage_settings[config.BATCH_SIZE_LIST]
    except Exception:
        batch_size_list = configs_count * [512]
    try:
        epochs_list = config.graphsage_settings[config.EPOCHS_LIST]
    except Exception:
        epochs_list = configs_count * [10]
    return num_walks_list, length_list, batch_size_list, epochs_list

def embeddings_hyperparameter_graphwave_gen(G, graphwave_settings_list, settings):
    """

    returns generator with node embeddings for GraphWave depending on its hyperparameter settings
    :param G (StellarGraph): similarity graph with node features
    :param graphwave_settings_list (list of Graphwave_settings): settings for hyperparameter tuning
    :param scaler (StandardScaler or MinMaxScaler): to scale node embeddings (with sklearn)
    :return: generator
    """
    if settings.graphwave_external_lib:
        gw_graph = get_graph_for_original_graphwave(G, settings)
    else:
        gw_graph = G
    for graphwave_settings in graphwave_settings_list:
        yield embeddings.generate_node_embeddings_graphwave(gw_graph, graphwave_settings, settings.scaler, settings.graphwave_external_lib)

def get_graph_for_original_graphwave(graph, settings):
    """
    creates or reads networkx similarity graph to use external GraphWave functionality
    :param graph (StellarGraph): similarity graph
    :param settings (Settings)
    :return: nx.Graph: similarity graph
    """
    if inout.graphwave_graph_exists(settings) and settings.mode == 'graph_load':
        G = inout.load_graph_for_graphwave_org(settings)
    else:
        G = StellarGraph.to_networkx(graph)
        inout.save_graph_for_graphwave_org(G, settings)
    return G

def get_default_params_graphwave(graphwave_ext, config):
    """
    sets (optional) values for graphwave hyperparameter tuning,
    default: degree = 30
    :param config (module): module where hyperparameter settings are set
    :return: list of Graphwave_settings
    """
    graphwave_settings_list = []
    sample_pct_list = config.graphwave_settings[config.SAMPLE_PCT_LIST]
    configs_count = len(sample_pct_list)
    try:
        degree = config.graphwave_settings[config.DEGREE_LIST]
    except:
        degree = configs_count * [30]
    for i in range(0, configs_count):
        sample_pct = config.graphwave_settings[config.SAMPLE_PCT_LIST][i]
        scales = config.graphwave_settings[config.SCALES_LIST][i]
        if not graphwave_ext and scales == ['auto']:
            scales = graphwave_ext.TAUS
        order_approx = degree[i]
        graphwave_settings = Graphwave_settings(scales=scales, sample_p_max_val=sample_pct[1], no_samples=sample_pct[2],
                                                order_approx=order_approx)
        graphwave_settings_list.append(graphwave_settings)
    return graphwave_settings_list



