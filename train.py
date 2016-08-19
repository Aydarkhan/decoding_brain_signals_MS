#!/usr/bin/python

import sys
sys.path.append('./bundle/')

import numpy as np
import io_dbs as io
from process import *
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV



try:
    import cPickle as pickle # Python 2 (cPickle = C version of pickle)
except ImportError:
    import pickle # Python 3 (C version is the default)


rndstate = 758494


filename = '../ecog_train_with_labels.csv'



def cross_validate(pdata, plabels, pen, params):
    accs = {}
    accs_tr = {}
    for p, data in pdata.iteritems():

        print "\n*** Patient %s ***\n" % (p)
        labels = plabels[p]

        sample_acc = []
        sample_acc_tr = []
        skf = StratifiedKFold(labels, 5)
        for train, test in skf:                                                             

            mc = MyClassifier(**params)
            mc.fit(data[train], labels[train], pen[p])

            pred = mc.predict(data[test])
            pred_tr = mc.predict(data[train])


            sample_acc.append(acc(labels[test], pred))
            sample_acc_tr.append(acc(labels[train], pred_tr))
            #sample_acc_tr.append(0)
            print "ACC ", sample_acc[-1]

        accs[p] = sample_acc
        accs_tr[p] = sample_acc_tr

    return accs_tr, accs

def save_classifiers(pdata, plabels, pen, params, fname=None, protocol=pickle.HIGHEST_PROTOCOL):
    print "\n*** Build classifiers ***\n"
    mc = {}
    if not fname:
        fname = '_'.join([k for k, v in sorted(params.iteritems()) if v]) + '.pkl'

    print "Classifier name: %s\n" % (fname)

    for p, data in pdata.iteritems():
        print "Patient %s" % p

        labels = plabels[p]

        mc[p] = MyClassifier(**params)
        mc[p].fit(data, labels, pen[p])

    with open('bundle/' + fname, 'wb') as fout:
            pickle.dump(mc, fout, protocol)

def save_stacked(pdata, plabels, sfname='stacked.pkl', protocol=pickle.HIGHEST_PROTOCOL):
    print "\n*** Build stacked ***\n"

    models = [
        dict(time=False, erbb=False, riemannERP=True, riemannXdawn=False, norm=True, featsel=False),
        dict(time=False, erbb=False, riemannERP=False, riemannXdawn=True, norm=False, featsel=False),
        dict(time=True, erbb=False, riemannERP=False, riemannXdawn=True, norm=True, featsel=True),
        dict(time=True, erbb=True, riemannERP=False, riemannXdawn=False, norm=True, featsel=False),
        dict(time=True, erbb=False, riemannERP=True, riemannXdawn=False, norm=True, featsel=True),
        dict(time=True, erbb=False, riemannERP=False, riemannXdawn=False, norm=True, featsel=False, bands=True),
        dict(time=True, erbb=True, riemannERP=False, riemannXdawn=False, norm=True, featsel=False, bands=True),
        dict(time=True, erbb=True, riemannERP=True, riemannXdawn=True, norm=True, featsel=True, bands=True)
    ]


    pen = {'p1': 10, 'p2': 10, 'p3': 10, 'p4':10}
    pen_erbb = {'p1': 0.1, 'p2': 0.1, 'p3': 0.3, 'p4':0.3} # TIME ERBB norm
    pen_xdawn = {'p1': 3, 'p2': 2000, 'p3': 1000, 'p4':1} #Riemann Xdawn

    sc = {}

    for p, data in pdata.iteritems():
        print "Patient %s" % p

        labels = plabels[p]

        sidx = shuffle(range(len(labels)), random_state=rndstate)
        sdata = data[sidx]
        slabels = labels[sidx]
        skf = StratifiedKFold(slabels, 5)

        predsf = []
        sclab = []
        for tridx, tstidx in skf:
            #print tstidx

            preds = []
            for params in models:
                pens = pen
                if params['time'] and params['erbb'] and not params['riemannXdawn']:
                    pens = pen_erbb
                if params['riemannXdawn'] and not params['time']:
                    pens = pen_xdawn

                mc = MyClassifier(**params)
                mc.fit(sdata[tridx], slabels[tridx], pen[p])
                proba = mc.predict_proba(sdata[tstidx])[:, 1:];
                preds.append(proba)
            preds = np.concatenate(preds, axis=1) #.astype(np.float32)
            predsf.append(preds)
            sclab.append(slabels[tstidx])

        predsf = np.concatenate(predsf, axis=0) #.astype(np.float32)
        sclab = np.concatenate(sclab) #.astype(np.float32)
        np.save(p + '_d.npy', predsf)
        np.save(p + '_l.npy', sclab)

        #predsf = np.load(p + '_d.npy')
        #sclab = np.load(p + '_l.npy')

        parameters = {'C':[0.001, 0.01, 0.1, 1., 10., 100., 1000.], 'gamma':[0.0001, 0.001, 0.01, 0.1, 1]}
        svr = SVC()
        clf = GridSearchCV(svr, parameters, cv=10)
        clf.fit(predsf, sclab)
        print clf.best_params_
        print clf.best_score_
        print clf.best_estimator_

        sc[p] = clf.best_estimator_
 

    with open('bundle/' + sfname, 'wb') as fout:
            pickle.dump(sc, fout, protocol)



if __name__ == '__main__':
    data = io.read_data(filename)
    pdata, pids, plabels = io.extract_data(data)

    pen = {'p1': 10, 'p2': 10, 'p3': 10, 'p4':10}

    params = dict(time=False, erbb=False, riemann=True, norm=True, featsel=False)


    ### Assess the model with cross-validation

    #accs_tr, accs = cross_validate(pdata, plabels, pen, params)
    #print "\n* Accuracies: \n"
    #print [(p, np.mean(i)) for p, i in accs.iteritems()]
    #print "\n* Total accuracy: ", np.mean([np.mean(i) for p, i in accs.iteritems()])


    ### Save different models, selected with cross-validation

    #params = dict(time=True, erbb=False, riemannERP=True, riemannXdawn=False, norm=True, featsel=True)
    #save_classifiers(pdata, plabels, pen, params, fname=None)


    #pen_tmp = {'p1': 0.1, 'p2': 0.1, 'p3': 0.3, 'p4':0.3} # TIME ERBB norm
    #params = dict(time=True, erbb=True, riemannERP=False, riemannXdawn=False, norm=True, featsel=False)
    #save_classifiers(pdata, plabels, pen_tmp, params, fname=None)

    #params = dict(time=False, erbb=False, riemannERP=True, riemannXdawn=False, norm=True, featsel=False)
    #save_classifiers(pdata, plabels, pen, params, fname=None)

    #pen_tmp = {'p1': 3, 'p2': 2000, 'p3': 1000, 'p4':1} #Riemann Xdawn
    #params = dict(time=True, erbb=False, riemannERP=False, riemannXdawn=True, norm=True, featsel=True)
    #save_classifiers(pdata, plabels, pen_tmp, params, fname=None)

    #params = dict(time=True, erbb=False, riemannERP=False, riemannXdawn=False, norm=True, featsel=False, bands=True)
    #save_classifiers(pdata, plabels, pen, params, fname=None)

    #params = dict(time=True, erbb=True, riemannERP=False, riemannXdawn=False, norm=True, featsel=False, bands=True)
    #save_classifiers(pdata, plabels, pen, params, fname=None)

    #params = dict(time=True, erbb=True, riemannERP=True, riemannXdawn=True, norm=True, featsel=True, bands=True)
    #save_classifiers(pdata, plabels, pen, params, fname=None)

    #params = dict(time=False, erbb=False, riemannERP=False, riemannXdawn=True, norm=False, featsel=False)
    #save_classifiers(pdata, plabels, pen, params, fname=None)

    ### Save stacked model
    #save_stacked(pdata, plabels, sfname='stacked.pkl')

