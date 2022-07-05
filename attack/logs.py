import numpy as np


def log_blk_dist(blk_dicts):
    blk_sizes = np.array([])
    for blk_dict in blk_dicts:
        for k in blk_dict:
            blk_sizes = np.append(blk_sizes, len(blk_dict[k]))
    print('No. of blocks: {}, Min: {}, Max: {}, Avg: {}, Med: {}'.format(len(blk_sizes), min(blk_sizes), max(blk_sizes), np.average(blk_sizes), np.median(blk_sizes)))