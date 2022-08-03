class graphfilesettings:
    def __init__(self, graph_filename):
        # pattern: c{0}_t{1}_r{2}{3}{4} OR c{0}_t{1}_r{2}_lshc{3}_lshs{4}{5}{6}
        file_without_extension = graph_filename[:graph_filename.rfind('.')]
        name_parts = file_without_extension.split("_")
        self.record_count = name_parts[0][1:]
        self.threshold = name_parts[1][1:]
        self.removed_records = name_parts[2][1:]
        if len(name_parts) > 4:
            self.lsh_count_blk = name_parts[3][4:]
            self.lsh_size_blk = name_parts[4][4:]
        if "histo" in name_parts[-1]:
            self.mode = "histo"
        elif "fast" in name_parts[-1]:
            self.mode = "fast"
        else:
            self.mode = "normal"