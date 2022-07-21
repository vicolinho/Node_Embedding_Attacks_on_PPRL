from attack import embeddings
from classes.deepgraphinfomax_settings import Deepgraphinfomax_settings
from classes.graphsage_settings import Graphsage_settings
import config.config as config
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

def get_default_params_deepgraphinfomax():
    layer_structures_list = config.deepgraphinfomax_settings[config.LAYER_STRUCTURES_LIST]
    activation_func_list = config.deepgraphinfomax_settings[config.ACTIVATION_FUNC_LIST]
    deepgraphinfomax_settings_list = []
    for i in range(0, len(layer_structures_list)):
        deepgraphinfomax_settings = Deepgraphinfomax_settings(layers=layer_structures_list[i], activations=activation_func_list[i])
        deepgraphinfomax_settings_list.append(deepgraphinfomax_settings)
    return deepgraphinfomax_settings_list

def embeddings_hyperparameter_graphsage(G, graphsage_settings_list, learning_G = None):
    embedding_results_list = []
    for graphsage_settings in graphsage_settings_list:
        embedding_results = embeddings.generate_node_embeddings_graphsage(G, graphsage_settings, learning_G)
        embedding_results_list.append(embedding_results)
    return embedding_results_list

def embeddings_hyperparameter_graphsage_gen(G, graphsage_settings_list, learning_G = None):
    for graphsage_settings in graphsage_settings_list:
        yield embeddings.generate_node_embeddings_graphsage(G, graphsage_settings, learning_G)

def get_default_params_graphsage():
    layer_structures_list = config.graphsage_settings[config.LAYER_STRUCTURES_LIST]
    num_samples_list = config.graphsage_settings[config.NUM_SAMPLES_LIST]
    graphsage_settings_list = []
    for i in range(0, len(layer_structures_list)):
        graphsage_settings = Graphsage_settings(layers=layer_structures_list[i], num_samples=num_samples_list[i])
        graphsage_settings_list.append(graphsage_settings)
    return graphsage_settings_list

def embeddings_hyperparameter_graphwave_gen(G, graphwave_settings_list):
    for graphwave_settings in graphwave_settings_list:
        yield embeddings.generate_node_embeddings_graphwave(G, graphwave_settings)

def get_default_params_graphwave(graphwave_lib_path):
    graphwave_settings_list = []
    degree_list = config.graphwave_settings[config.DEGREE_LIST]
    for i in range(0, len(degree_list)):
        sample_pct = config.graphwave_settings[config.SAMPLE_PCT_LIST][i]
        scales = config.graphwave_settings[config.SCALES_LIST][i]
        graphwave_settings = Graphwave_settings(graphwave_libpath=graphwave_lib_path,
            scales=scales, sample_p_max_val=sample_pct[1], no_samples=sample_pct[2], degree=degree_list[i]
        )
        graphwave_settings_list.append(graphwave_settings)
    return graphwave_settings_list



