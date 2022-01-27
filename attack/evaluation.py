from pathlib import Path

def evalaute_top_pairs(matches, true_matches):
    counter = 0
    for match in matches:
        if match in true_matches:
            counter += 1
    return counter / len(matches)

def output_result(technique, prec, output_path, record_count, threshold, removed_frac):
    csv_header = "technique,prec\n"
    filename = "c{0}_t{1}_r{2}.csv".format(record_count, threshold, removed_frac)
    output_path = output_path + "/" if output_path[-1] != "/" else output_path
    full_path = output_path + filename
    header_needed = not Path(full_path).is_file()
    with open(full_path, 'a') as file:
        if header_needed:
            file.write(csv_header)
        file.write(technique + "," + str(prec) + "\n")

