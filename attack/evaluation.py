def evalaute_top_pairs(matches, true_matches):
    counter = 0
    for match in matches:
        if match in true_matches:
            counter += 1
    return counter / len(matches)