# possible to import config from path -> https://docs.python.org/3/library/importlib.html?highlight=import_module#importing-a-source-file-directly

LAYER_STRUCTURES_LIST = 'LAYER_STRUCTURES_LIST'
NUM_SAMPLES_LIST = 'NUM_SAMPLES_LIST'
ACTIVATION_FUNC_LIST = 'ACTIVATION_FUNC_LIST'

graphsage_settings = {
    LAYER_STRUCTURES_LIST: [[128,128], [128,128], [256, 128]],
    NUM_SAMPLES_LIST: [[10,5], [20,20], [10, 5]]
}

deepgraphinfomax_settings = {
    LAYER_STRUCTURES_LIST: [[512,256,128], [128,128,128,128], [128], [1024,1024,128]],
    ACTIVATION_FUNC_LIST: [3 * ['relu'], 4 * ['relu'], 1 * ['relu'], 3 * ['relu']]
}