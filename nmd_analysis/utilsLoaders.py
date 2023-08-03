
from pathlib import Path

import numpy as np
import pandas as pd


def read_trc(fpath):
    # read metadata in file header
    df_meta = pd.read_csv(fpath, delimiter='\t', header=0, skiprows=1, nrows=1)
    meta = df_meta.iloc[0].to_dict()
    fps = meta['DataRate']

    # read marker location names
    markers_df = pd.read_csv(fpath, delimiter='\t', header=None, skiprows=3, nrows=1)
    markers = markers_df.iloc[0].dropna().to_numpy()[2:]

    # read marker XYZ locations
    df = pd.read_csv(fpath, delimiter='\t', header=0, skiprows=4)
    df.rename(columns=dict(zip(df.columns[:2], ('n', 't'))), inplace=True)
    df.dropna(how='all', axis=1, inplace=True)

    N = df.shape[0]
    M = len(markers)
    xyz = df.iloc[:,2:].to_numpy().reshape((N, M, 3))
    xyz[:,:,[0,1,2]] = xyz[:,:,[2,1,0]]

    return fps, markers, xyz


def read_mot(fpath):
    with open(fpath, 'r') as f:
        count = 0
        line = f.readline()
        while line and line.strip() != 'endheader':
            line = f.readline().strip()
            count += 1

        df = pd.read_csv(f, delimiter='\t', header=0, skiprows=count-1)

    return df


