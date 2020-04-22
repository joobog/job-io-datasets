#!/usr/bin/env python3

import os
import sys
import glob
import time
import multiprocessing as mp
import pandas as pd
import numpy as np

# from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn import tree

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=3)


class Config:
    '''Prediction'''
    def __init__(
            self,
            input_csv,
            metadata_csv,
            cluster_csv,
            label_csv,
            train_size,
            columns,
            dists):
        self.input_csv = input_csv
        self.metadata_csv = metadata_csv
        self.cluster_csv = cluster_csv
        self.label_csv = label_csv
        self.columns = columns,
        self.train_size = train_size,
        self.dists = dists


def cluster(data, dist, cfg):
    '''Prediction'''
    cols = cfg.columns[0]
    cluster_data = data[cols]
    cluster_data = cluster_data[0:cfg.train_size[0]]
    X = Normalizer().fit_transform(cluster_data)
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=dist)
    model.fit(X)
    y_pred = model.labels_.astype(np.int)
    cluster_data['cluster'] = y_pred
    cluster_data['dist'] = dist
    # cluster_data.reset_index(inplace=True)
    # cluster_data.set_index(['jobid', 'dist'], inplace=True)
    return cluster_data


if __name__ == '__main__':
    # Configuration at on place

    METRICS = ['md_file_create', 'md_file_delete', 'md_mod', 'md_other', 'md_read', 'read_bytes', 'read_calls', 'write_bytes', 'write_calls']
    COLUMNS = [str(n) + '_' + metric for n in [1, 4] for metric in METRICS]

    CFG = Config(
        input_csv='../../datasets/job_io_duration.csv',
        metadata_csv='../../datasets/job_metadata.csv',
        cluster_csv='../../evaluation/job_io_duration_clustered.csv',
        label_csv='../../evaluation/job_io_duration_labeled.csv',
        train_size=10000,
        columns=COLUMNS,
        dists=list(np.arange(0.1, 1, 0.1))
        )


    # Read and prepare input
    DATA = pd.read_csv(CFG.input_csv, index_col='jobid', dtype={'jobid':np.int64})
    print(DATA.head())
    METADATA = pd.read_csv(CFG.metadata_csv, index_col='jobid', dtype={'jobid':np.int64, 'utilization':np.float})
    print(METADATA.head())
    DATA = pd.merge(DATA, METADATA, left_on='jobid', right_on='jobid')
    print("DATA", DATA.head())


    # Clustering
    # with hierachical algorithm
    # with several distances
    print('Clustering')
    DATA.dropna(inplace=True)
    RES = list()
    for dist in CFG.dists:
        start = time.time()
        RES.append(cluster(DATA, dist, CFG))
        stop = time.time()
        print('Duration %f seconds' % (stop - start))
    RES_DF = pd.concat(RES)
    RES_DF.to_csv(CFG.cluster_csv)



    # Classification
    # with decision trees

    os.remove(CFG.label_csv)

    for dist in CFG.dists:
        print('Classification %f' % (dist))
        RES_DF = RES_DF[RES_DF['dist'] == dist]
        X = RES_DF[CFG.columns[0]]
        Y = RES_DF['cluster']
        CLF = tree.DecisionTreeClassifier()
        CLF = CLF.fit(X, Y)

        LABELS = CLF.predict(DATA[CFG.columns[0]])
        DATA['labels'] = LABELS
        if os.path.exists(CFG.label_csv):
            DATA.to_csv(CFG.label_csv, mode='a', header=False)
        else:
            DATA.to_csv(CFG.label_csv, mode='w', header=True)

