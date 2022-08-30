class graphfilesettings:
    def __init__(self, graph_filename):
        # pattern: c{0}_t{1}_r{2}e{3}{4}{5} OR c{0}_t{1}_r{2}e{3}_lshc{4}_lshs{5}{6}{7}
        file_without_extension = graph_filename[:graph_filename.rfind('.')]
        name_parts = file_without_extension.split("_")
        self.record_count = name_parts[0][1:]
        self.threshold = name_parts[1][1:]
        self.set_removed_records(name_parts[2])
        if len(name_parts) > 4:
            self.lsh_count_blk = name_parts[3][4:]
            self.lsh_size_blk = name_parts[4][4:]
        if "histo" in name_parts[-1]:
            self.mode = "histo"
        elif "fast" in name_parts[-1]:
            self.mode = "fast"
        else:
            self.mode = "normal"

    def set_removed_records(self, removed_records_namepart):
        e_pos = removed_records_namepart.find("e")
        if e_pos == -1:
            self.removed_records_plain = float(removed_records_namepart[1:])
            self.removed_records_encoded = 0.0
        else:
            self.removed_records_plain = float(removed_records_namepart[1:e_pos])
            self.removed_records_encoded = float(removed_records_namepart[e_pos+1:])