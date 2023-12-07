# this is a config file for hyperparameter tuning with the different node embedding techniques GraphSAGE, DeepGraphInfomax, GraphWave

# you can edit the values of the settings dicts here in this file
# or you can make a copy in the same directory
# if you want to use the copy you have to include it in your programm call with the option hp_config_file
# for a file config_copy.py i.e. --hp_config_file config_copy

# needed params for deepgraphinfomax + graphsage hyperparameter tuning
LAYER_STRUCTURES_LIST = 'LAYER_STRUCTURES_LIST' # list of lists of layer sizes
EPOCHS_LIST = 'EPOCHS_LIST' # list

# needed params for graphsage hyperparameter tuning
NUM_SAMPLES_LIST = 'NUM_SAMPLES_LIST' # list of lists
NUM_WALKS_LIST = 'NUM_WALKS_LIST' # list of lists
LENGTH_LIST = 'LENGTH_LIST' #
BATCH_SIZE_LIST = 'BATCH_SIZE_LIST'

# needed params for deepgraphinfomax hyperparameter tuning
ACTIVATION_FUNC_LIST = 'ACTIVATION_FUNC_LIST' # list of lists

#needed params for graphwave hyperparameter tuning
SAMPLE_PCT_LIST = 'SAMPLE_PCT_LIST'
SCALES_LIST = 'SCALES_LIST'
DEGREE_LIST = 'DEGREE_LIST'

# all values in the dictionary have to be lists, one element stands for one hyperparameter run

graphsage_settings = {
    LAYER_STRUCTURES_LIST: [[256,128]],
    NUM_SAMPLES_LIST: [[10,5]],
    EPOCHS_LIST: [1]
}

deepgraphinfomax_settings = {
    LAYER_STRUCTURES_LIST: [[512]],
    ACTIVATION_FUNC_LIST: [['relu']],
    EPOCHS_LIST: [20]
}

graphwave_settings = {
    SAMPLE_PCT_LIST: [(0,100,50)],
    SCALES_LIST: [['auto']]
}