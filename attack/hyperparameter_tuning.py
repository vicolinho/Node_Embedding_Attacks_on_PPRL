from attack import embeddings
from classes.deepgraphinfomax_settings import Deepgraphinfomax_settings
from classes.graphsage_settings import Graphsage_settings

LAYER_STRUCTURES_LIST = [[512,256,128], [128,128,128,128], [128], [1024,1024,128]]
ACTIVATION_FUNC_LIST = [3 * ['relu'], 4 * ['relu'], 1 * ['relu'], 3 * ['relu']]

def embeddings_hyperparameter_deepgraphinfomax(G, deepgraphinfomax_settings_list):
    embedding_results_list = []
    for deepgraphinfomax_settings in deepgraphinfomax_settings_list:
        embedding_results = embeddings.generate_node_embeddings_deepgraphinfomax(G, deepgraphinfomax_settings)
        embedding_results_list.append(embedding_results)
    return embedding_results_list

def get_default_params_deepgraphinfomax():
    layer_structures_list = LAYER_STRUCTURES_LIST
    activation_func_list = ACTIVATION_FUNC_LIST
    deepgraphinfomax_settings_list = []
    for i in range(0, len(layer_structures_list)):
        deepgraphinfomax_settings = Deepgraphinfomax_settings(layers=layer_structures_list[i], activations=activation_func_list[i])
        deepgraphinfomax_settings_list.append(deepgraphinfomax_settings)
    return deepgraphinfomax_settings_list

def embeddings_hyperparameter_graphsage(G, graphsage_settings_list, learning_G = None):
    embedding_results_list = []
    for graphsage_settings in graphsage_settings_list:
        info_string = str(graphsage_settings)
        embedding_results = embeddings.generate_node_embeddings_graphsage(G, graphsage_settings, learning_G)
        embedding_results_list.append(embedding_results)
    return embedding_results_list

def get_default_params_graphsage():
    layer_structures_list = [[128,128], [128,128]]#, [256, 128], [512, 256, 128]]
    num_samples_list = [[10,5], [20,20]]#, [10, 5], [30, 20, 10]]
    graphsage_settings_list = []
    for i in range(0, len(layer_structures_list)):
        graphsage_settings = Graphsage_settings(layers=layer_structures_list[i], num_samples=num_samples_list[i])
        graphsage_settings_list.append(graphsage_settings)
    return graphsage_settings_list



