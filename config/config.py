# possible to import config from path -> https://docs.python.org/3/library/importlib.html?highlight=import_module#importing-a-source-file-directly

LAYER_STRUCTURES_LIST = 'LAYER_STRUCTURES_LIST'
NUM_SAMPLES_LIST = 'NUM_SAMPLES_LIST'
ACTIVATION_FUNC_LIST = 'ACTIVATION_FUNC_LIST'
SAMPLE_PCT_LIST = 'SAMPLE_PCT_LIST'
SCALES_LIST = 'SCALES_LIST'
DEGREE_LIST = 'DEGREE_LIST'


graphsage_settings = {
    LAYER_STRUCTURES_LIST: [[128,128], [128,128], [256, 128]],
    NUM_SAMPLES_LIST: [[10,5], [20,20], [10, 5]]
}

deepgraphinfomax_settings = {
    LAYER_STRUCTURES_LIST: [[512,256,128], [128,128,128,128], [128], [1024,1024,128]],
    ACTIVATION_FUNC_LIST: [3 * ['relu'], 4 * ['relu'], 1 * ['relu'], 3 * ['relu']]
}

graphwave_settings = {
    SAMPLE_PCT_LIST: [(0,100,50), (0,50,25), (0,100,50), (0,50,25)],
    SCALES_LIST: 4 * [[5,10]],
    DEGREE_LIST: [20,20,30,30]
}