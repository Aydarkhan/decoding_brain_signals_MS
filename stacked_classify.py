#!/usr/bin/python

import sys
if ".\Script Bundle" not in sys.path:
    sys.path.append('./bundle/')

import io_dbs as io
import numpy as np
import pandas as pd

from process import *

try:
    import cPickle as pickle 
except ImportError:
    import pickle 


filename = '../ecog_train_with_labels.csv'


def load_classifiers(fname='ga.pkl'):

    with open(fname, 'rb') as fin:
       return pickle.load(fin)

def azureml_main(dataframe1 = None, dataframe2 = None):
    pdata, pids, plabels = io.extract_data(dataframe1)
    cls = [
        'norm_riemannERP.pkl',
        'riemannXdawn.pkl',
        'featsel_norm_riemannXdawn_time.pkl',
        'erbb_norm_time.pkl',
        'featsel_norm_riemannERP_time.pkl',
        'bands_norm_time.pkl',
        'bands_erbb_norm_time.pkl',
        'bands_erbb_featsel_norm_riemannERP_riemannXdawn_time.pkl',
        ]

    scn = 'stacked.pkl'
    mcs = {}
    for fname in cls:
        if ".\Script Bundle" in sys.path:
            fname = ".\Script Bundle\\" + fname
        else:
            fname = "bundle/" + fname
        mcs[fname] = load_classifiers(fname)

    fname = scn
    if ".\Script Bundle" in sys.path:
        fname = ".\Script Bundle\\" + fname
    else:
        fname = "bundle/" + fname
    sc = load_classifiers(fname)

    res = []
    for p, data in pdata.iteritems():

        preds = []
        for mc in mcs.values():
            preds.append(mc[p].predict_proba(data)[:, 1:])
        preds = np.concatenate(preds, axis=1)
        pred = sc[p].predict(preds)

        res.append(
                pd.DataFrame(pd.Series([p]*data.shape[0], name='PatientID'))
                )
        res[-1]['Stimulus_ID'] = pd.Series(pids[p], index=res[-1].index)
        res[-1]['Scored Labels'] = pd.Series(pred, index=res[-1].index)

    return pd.concat(res),

if __name__ == '__main__':
    data = io.read_data(filename)
    out, = azureml_main(dataframe1 = data)


