from attack import embeddings
from classes.deepgraphinfomax_settings import Deepgraphinfomax_settings
from classes.graphsage_settings import Graphsage_settings
import config.config as config

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



