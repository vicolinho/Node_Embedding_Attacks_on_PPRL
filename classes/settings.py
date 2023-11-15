import importlib

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Settings:
    def __init__(self, parser):
        self.mode = parser.mode
        if self.mode == "graph_save" or self.mode == "graph_save_calculate":
            self.record_count = int(parser.record_count)
            self.threshold = float(parser.threshold)
            self.removed_plain_record_frac = float(parser.remove_frac_plain)
            self.removed_encoded_record_frac = float(parser.remove_frac_encoded)
            self.node_features = parser.node_features
            self.node_count = parser.node_count
            self.graph_path = parser.graph_path
            self.plain_file = parser.plain_file
            self.encoded_file = parser.encoded_file
            self.graph_scaled = parser.graph_scaled
            self.padding = parser.padding
            self.lsh_count_blk = int(parser.lsh_count_blocking)
            self.lsh_size_blk = int(parser.lsh_size_blocking)
            self.qgram_attributes = parser.qgram_attributes
            self.encoded_attr = parser.encoded_attr

        if self.mode == "graph_load" or self.mode == "graph_save_calculate":
            self.results_path = parser.results_path
            self.graphwave_external_lib = not parser.graphwave_sg_lib
            self.weights = [[w, 1 - w] for w in parser.weight_list]
            self.hp_config = importlib.import_module("config." + parser.hp_config_file)
            self.cos_sim_thold = float(parser.node_matching_threshold)
            self.vidanage_weights = parser.vidanage_weights
            self.normalize_weights_vidanage()
            self.lsh_count_nm = int(parser.lsh_count_node_matching)
            self.lsh_size_nm = int(parser.lsh_size_node_matching)
            self.hyperplane_count = int(1024)
            self.num_top_pairs = parser.num_top_pairs
            self.node_matching_tech = parser.node_matching_tech
            if parser.scaler == 'standardscaler':
                self.scaler = StandardScaler()
            elif parser.scaler == 'minmaxscaler':
                self.scaler = MinMaxScaler()

        if self.mode == "graph_load":
            self.pickle_file = parser.pickle_file

        self.min_comp_size = int(parser.min_comp_size)

    def normalize_weights_vidanage(self):
        w_cos, w_sim_conf, w_degr_conf = self.vidanage_weights[0], self.vidanage_weights[1], self.vidanage_weights[2]
        sum_weights = w_cos + w_sim_conf + w_degr_conf
        if sum_weights != 1.0:
            w_cos /= sum_weights
            w_sim_conf /= sum_weights
            w_degr_conf /= sum_weights
            self.vidanage_weights = [w_cos, w_sim_conf, w_degr_conf]

    def is_gw_lib_valid(self):
        try:
            import sys
            sys.path.insert(0, self.graphwave_external_lib)
            import graphwave
            return True
        except:
            return False