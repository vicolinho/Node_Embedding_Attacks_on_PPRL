class Settings:
    def __init__(self, parser):
        self.results_path = parser.results_path
        self.record_count = parser.record_count
        self.threshold = parser.threshold
        self.removed_plain_record_frac = parser.remove_frac_plain
        self.histo_features = parser.histo_features
        self.lsh_count = parser.lsh_count
        self.lsh_size = parser.lsh_size
        self.cos_sim_thold = 0.3
        self.hyperplane_count = 1024
        self.num_top_pairs = 50
