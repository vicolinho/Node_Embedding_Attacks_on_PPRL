from attack.node_embeddings import embeddings
from classes.deepgraphinfomax_settings import Deepgraphinfomax_settings
from classes.graphsage_settings import Graphsage_settings
from classes.graphwave_settings import Graphwave_settings

def embeddings_hyperparameter_deepgraphinfomax(G, deepgraphinfomax_settings_list): # not in use because just combined with features which is better with generators but perhaps still needed
    embedding_results_list = []
    for deepgraphinfomax_settings in deepgraphinfomax_settings_list:
        embedding_results = embeddings.generate_node_embeddings_deepgraphinfomax(G, deepgraphinfomax_settings)
        embedding_results_list.append(embedding_results)
    return embedding_results_list

def embeddings_hyperparameter_deepgraphinfomax_gen(G, deepgraphinfomax_settings_list):
    for deepgraphinfomax_settings in deepgraphinfomax_settings_list:
        yield embeddings.generate_node_embeddings_deepgraphinfomax(G, deepgraphinfomax_settings)

def get_params_deepgraphinfomax(config):
    layer_structures_list = config.deepgraphinfomax_settings[config.LAYER_STRUCTURES_LIST]
    activation_func_list = config.deepgraphinfomax_settings[config.ACTIVATION_FUNC_LIST]
    epochs_list = get_default_params_deepgraphinfomax(config)
    deepgraphinfomax_settings_list = []
    for i in range(0, len(layer_structures_list)):
        deepgraphinfomax_settings = Deepgraphinfomax_settings(layers=layer_structures_list[i], activations=activation_func_list[i], epochs=epochs_list[i])
        deepgraphinfomax_settings_list.append(deepgraphinfomax_settings)
    return deepgraphinfomax_settings_list

def get_default_params_deepgraphinfomax(config):
    """ sets optional values for deepgraphinfomax hyperparameter tuning, default: epochs = 100 """
    configs_count = len(config.deepgraphinfomax_settings[config.LAYER_STRUCTURES_LIST])
    try:
        epochs_list = config.deepgraphinfomax_settings[config.EPOCHS_LIST]
    except Exception:
        epochs_list = configs_count * [100]
    return epochs_list

def embeddings_hyperparameter_graphsage(G, graphsage_settings_list, learning_G = None):
    embedding_results_list = []
    for graphsage_settings in graphsage_settings_list:
        embedding_results = embeddings.generate_node_embeddings_graphsage(G, graphsage_settings, learning_G)
        embedding_results_list.append(embedding_results)
    return embedding_results_list

def embeddings_hyperparameter_graphsage_gen(G, graphsage_settings_list, learning_G = None):
    for graphsage_settings in graphsage_settings_list:
        yield embeddings.generate_node_embeddings_graphsage(G, graphsage_settings, learning_G)

def get_params_graphsage(config):
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
    """ sets optional values for graphsage hyperparameter tuning, default: number_of_walks = 1, length = 5, batch_size = 50, epochs = 10 """
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
        batch_size_list = configs_count * [50]
    try:
        epochs_list = config.graphsage_settings[config.EPOCHS_LIST]
    except Exception:
        epochs_list = configs_count * [10]
    return num_walks_list, length_list, batch_size_list, epochs_list

def embeddings_hyperparameter_graphwave_gen(G, graphwave_settings_list):
    for graphwave_settings in graphwave_settings_list:
        yield embeddings.generate_node_embeddings_graphwave(G, graphwave_settings)

def get_default_params_graphwave(graphwave_lib_path, config):
    graphwave_settings_list = []
    sample_pct_list = config.graphwave_settings[config.SAMPLE_PCT_LIST]
    for i in range(0, len(sample_pct_list)):
        sample_pct = config.graphwave_settings[config.SAMPLE_PCT_LIST][i]
        scales = config.graphwave_settings[config.SCALES_LIST][i]
        graphwave_settings = Graphwave_settings(graphwave_libpath=graphwave_lib_path,
            scales=scales, sample_p_max_val=sample_pct[1], no_samples=sample_pct[2]
        )
        graphwave_settings_list.append(graphwave_settings)
    return graphwave_settings_list



