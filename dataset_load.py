import glob
import os
import time
import numpy as np
import pandas as pd
import pickle
import torch.nn as nn


def get_data_partition(partition_file):
    vid2partition, partition2vid = {}, {}
    df = pd.read_csv(partition_file)

    for row in df.values:

        vid, partition = str("%03d"%row[0]), row[1]
        vid2partition[vid] = partition
        if partition not in partition2vid:
            partition2vid[partition] = []
        if vid not in partition2vid[partition]:
            partition2vid[partition].append(vid)
    return vid2partition, partition2vid

def load_data(args):
    # Multiple feature path
    if isinstance(args.feature_set, list):
        feature_paths = [os.path.join(args.dataset_file_path, feat) for feat in args.feature_set]
        feature_set_name = '_'.join(args.feature_set)
    else:
        feature_paths = [os.path.join(args.dataset_file_path, args.feature_set)]
        feature_set_name = args.feature_set

    label_path = os.path.join(args.label_personality)
    partition_path = os.path.join(args.partition)

    data_file_name = f'data_{args.data_source}_{feature_set_name}.pkl'
    data_cache_path = "./data_cache/"
    os.makedirs(data_cache_path, exist_ok=True)
    data_file = os.path.join(data_cache_path, data_file_name)

    # Load directly if there are data_cache, 
    if os.path.exists(data_file):  
        print(f'Find cached data "{os.path.basename(data_file)}".')
        data = pickle.load(open(data_file, 'rb'))
        return data
    
    print('Constructing data from scratch ...')
    data = {'train': {'feature': [], 'label': []},
            'val': {'feature': [], 'label': []}}
    vid2partition, partition2vid = get_data_partition(partition_path)

    label_df = pd.read_csv(label_path)

    #Each feature serves as a sample
    for partition, vids in partition2vid.items():
        for vid in vids:
            current_features = []

            for feature_path in feature_paths:
                dir = os.path.join(feature_path, vid)
                for file in sorted(os.listdir(dir)):
                    if file.endswith("csv"):
                        feature = pd.read_csv(os.path.join(dir, file), header=None).to_numpy()
                    elif file.endswith("npy"):
                        feature = np.load(os.path.join(dir, file))

                    # If it is a two-dimensional feature, perform max pooling    
                    if feature.ndim == 2:
                        feature = np.max(feature, axis=0, keepdims=True)
                    feature = np.nan_to_num(feature, nan=0.0)
                    current_features.append(feature)


            if current_features:
                combined_feature = np.concatenate(current_features, axis=1)

                label_row = label_df[label_df['id'] == int(vid)]
                if label_row.empty:
                    continue
                label = label_row.iloc[0, 1:].to_numpy()

                data[partition]['label'].append(label)
                data[partition]['feature'].append(combined_feature)

    # Save data_cache
    if not os.path.exists("./data_cache"):
        os.mkdir("./data_cache")
    pickle.dump(data, open(data_file, 'wb'))


    return data

