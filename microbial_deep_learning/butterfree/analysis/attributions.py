import numpy as np 
from collections import defaultdict 

def significant_features(attribution, threshold, column_names=None, label_names=None):
    """
    Given a BxLxC tensor, where B = batch dimension, L = labels dimension, C = columns dimension,
    Returns the column names where the value is above a certain threshold for each row. 
    """
    above_threshold = [np.where([y>threshold for y in attribution[i]]) for i in range(len(attribution))]
    features_dicts = [] 
    def map_columns(j):
        if column_names is None:
            return j 
        else:
            return column_names[j] 
    def map_labels(i):
        if label_names is None:
            return i
        else:
            return label_names[i]

    for example in above_threshold:
        f_d = defaultdict(lambda: [])
        for label ,feature in zip(*example):
            f_d[map_labels(label)].append(map_columns(feature))
        features_dicts.append(dict(f_d))

    return features_dicts

def aggregate(features_dicts):
    """
    Combines features dicts into a single dict mapping labels to every feature associated with them. 
    """    
    # Remove empty dicts 
    features_dicts = [x for x in features_dicts if x != {}]

    aggregate_dict = defaultdict(lambda: [])
    for f in features_dicts:
        key = list(f.keys())[0]
        value = list(f.values())[0]
        aggregate_dict[key].append(value)

    return aggregate_dict

def consolidate(aggregate_dict):
    """

    """
    return {key: list(set([x for y in value for x in y])) for key, value in aggregate_dict.items()}