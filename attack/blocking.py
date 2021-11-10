from pandas import DataFrame

def blk_func(func, *args):
    key = str()
    for arg in args:
        key += func(arg)
    return key

sound_dict = {
    **dict.fromkeys(['a', 'e', 'i', 'o', 'u', 'y', 'h', 'w'], ''),
    **dict.fromkeys(['b', 'f', 'p', 'v'], '1'),
    **dict.fromkeys(['c', 'g', 'j', 'k', 'q', 's', 'x', 'z'], '2'),
    **dict.fromkeys(['d', 't'], '3'),
    **dict.fromkeys(['l'], '4'),
    **dict.fromkeys(['m', 'n'], '5'),
    **dict.fromkeys(['r'], '6')
  }

def soundex(attr):
    if len(attr) == 0:
        return '0000'
    elif len(attr) == 1:
        start, attr = attr[0], ''
    else:
        start, attr = attr[0], attr[1:]
    soundex_attr = ''
    last_number = -1
    for attr_char in attr:
        if not attr_char in sound_dict.keys():
            continue
        curr_number = sound_dict[attr_char]
        if curr_number != last_number:
            soundex_attr += curr_number
            last_number = curr_number if curr_number != '' else last_number
        if len(soundex_attr) >= 3:
            break
    while len(soundex_attr) < 3:
        soundex_attr += '0'
    soundex_attr = start + soundex_attr
    return soundex_attr

def soundex_list(attr_list):
    sdxcode = str()
    for attr in attr_list:
        sdxcode += soundex(attr)
    return sdxcode

def get_blocking_dict(df):
    blk_dict = dict()
    grouped_blk_df = df.groupby("blocking")
    for blk_key in df['blocking'].unique():
        blk_dict[blk_key] = grouped_blk_df.get_group(blk_key)
    return blk_dict

def add_blocking_keys_to_dfs(df_all, blk_attr):
    cols = DataFrame()
    for attribute in blk_attr:
        cols[attribute] = df_all[attribute]
    return cols.apply(func=soundex_list, axis=1)