# deepgraphinfomax + graphsage
LAYER_STRUCTURES_LIST = 'LAYER_STRUCTURES_LIST'
EPOCHS_LIST = 'EPOCHS_LIST'

#graphsage
NUM_SAMPLES_LIST = 'NUM_SAMPLES_LIST'
NUM_WALKS_LIST = 'NUM_WALKS_LIST'
LENGTH_LIST = 'LENGTH_LIST'
BATCH_SIZE_LIST = 'BATCH_SIZE_LIST'

#deepgraphinfomax
ACTIVATION_FUNC_LIST = 'ACTIVATION_FUNC_LIST'

#graphwave
SAMPLE_PCT_LIST = 'SAMPLE_PCT_LIST'
SCALES_LIST = 'SCALES_LIST'
DEGREE_LIST = 'DEGREE_LIST'


graphsage_settings = {
    LAYER_STRUCTURES_LIST: [[256,128]],#[[128,128], [128,128], [256, 128]],
    NUM_SAMPLES_LIST: [[10,5]],#[[10,5], [20,20], [10, 5]]
    EPOCHS_LIST: [1]
}

deepgraphinfomax_settings = {
    LAYER_STRUCTURES_LIST: [[512,256,128]],#[[512,256,128], [128,128,128,128], [128], [1024,1024,128]],
    ACTIVATION_FUNC_LIST: [3 * ['relu']],#[3 * ['relu'], 4 * ['relu'], 1 * ['relu'], 3 * ['relu']]
    EPOCHS_LIST: [1]
}

graphwave_settings = {
    SAMPLE_PCT_LIST: [],#[(0,100,50), (0,50,25), (0,100,50), (0,50,25)],
    SCALES_LIST: []#2 * [['auto']] + 2 * [[5,10]]
}