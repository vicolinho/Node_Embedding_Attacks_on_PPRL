def evalaute_top_pairs(matches, true_matches):
    if len(matches) == 0:
        return float('nan')
    counter = 0
    for match in matches:
        if match in true_matches:
            counter += 1
    return counter / len(matches)

