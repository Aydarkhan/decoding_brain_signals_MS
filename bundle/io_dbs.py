#!/usr/bin/python

import pandas as pd
import numpy as np
import sys

def read_data(filename):
    data = pd.read_csv(filename)
    return data


def extract_data(data):
    data = data[data.Stimulus_ID != -1]

    pdata = {}
    pids = {}
    plabels = {}
    for p in data.PatientID.unique():
        df = data[data.PatientID == p]

        # Remove bad channels
        df.drop(df.columns[(df == -999999).all()],axis=1,inplace=True)
        
        st = df.Stimulus_Type.as_matrix()

        if 0 < st[-1] < 101: st = np.concatenate([st, [101]]) # to properly deal with the end condition for the next operation
        lastid = np.where(np.diff(st) > 0 )[0] # get last index of each stimulus
        labels = np.floor((st[lastid] - 1) / 50.0).astype(np.int64) + 1 # 1 or 2

        ids = df.Stimulus_ID.as_matrix()[lastid - 200]  # sample ID, a value from the middle of a window (random)

        df.drop(['Stimulus_ID', 'Stimulus_Type', 'PatientID'], axis=1, inplace=True)

        #print "ids", ids
        #print "labels ", labels

        # Arrange data in 3D tensor m
        m = []
        dmatrix = df.as_matrix()
        for i in lastid:
            m.append(dmatrix[i-799:i+1,:])

        sdata = np.array(m) * 1e-4

        pdata[p] = sdata
        pids[p] = ids
        plabels[p] = labels

    return pdata, pids, plabels

