import numpy as np
from scipy.signal.filter_design import butter
from scipy.signal import filtfilt, welch
from mne.time_frequency import cwt_morlet
from sklearn.decomposition import PCA

from pyriemann.estimation import XdawnCovariances, ERPCovariances
from pyriemann.tangentspace import TangentSpace

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif



# Sampling frequency
Fs = 1000

def preprocess(data):
    ''' 3D (trial, time, channel) '''
    res = data - data.mean(axis=(0, 1))
    return res

def filter(data):
    '''
    Butterworth bandpass filter and sub-sampling
    '''
    # Filter
    ord = 4
    wn = np.array([1., 10.]) / Fs * 2
    b, a = butter(ord, wn, btype='bandpass')
    res = filtfilt(b,a, data, axis=1)
    res = res[:,-450:-60,:]
    #res = res[:,-450:-60:10,:]
    return res

def get_bands(data):
    '''
    Complex morlet continuous wavelet transform from 4 to 10 Hz
    '''

    ord = 4
    wn = 10. / Fs * 2
    b, a = butter(ord, wn, btype='lowpass')

    erb = []

    freqs=np.arange(4,11,2)
    n_cycles= [2,3] + 2*[4]
    
    for d in data:
        ft = cwt_morlet(d, sfreq=Fs, freqs=freqs, n_cycles=n_cycles)
        trft = ft.transpose((2,0,1))
        powft = np.log(np.abs(trft))
        fbb = filtfilt(b,a, powft, axis=0).reshape((trft.shape[0],-1))
        erb.append(fbb)
    erb = np.array(erb)
    erb = erb[:,-450:-60,:]
    #erb = erb[:,-450:-60:10,:]

    return erb



def get_spc(data):
    ''' data -> (epochs, channels, time) 
    
    Get parameters for PCA
    '''
    Ps = []
    for d in data:
        _, P = welch(d[:,-400:], fs=Fs, nperseg=200)
        Ps.append(P)
    for d in data:
        _, P = welch(d[:,:400], fs=Fs, nperseg=200)
        Ps.append(P)

    Ps = np.array(Ps)
    keepPs = Ps.copy()
    Ps = np.log(Ps)
    m = Ps.mean(axis=0)
    Ps -= np.repeat(m[np.newaxis,...], len(Ps), axis=0)
    pca = []
    for c in range(Ps.shape[1]):
        pca.append( PCA(1))
        pca[-1].fit(Ps[:,c,14:50])
    return pca, m

def get_erbb_mor(data, pca, m):
    '''
    ERBB estimation via morlet wavelet and PCA parameters
    '''

    ord = 4
    wn = 20. / Fs * 2
    b, a = butter(ord, wn, btype='lowpass')

    erbb = []

    freqs=np.arange(70,250,5)
    n_cycles=7

    for d in data:
        ft = cwt_morlet(d, sfreq=Fs, freqs=freqs, n_cycles=n_cycles)
        trft = ft.transpose((2,0,1))
        powft = np.abs(trft) ** 2
        nft =  powft 
        logft = np.log(nft)
        logft -= np.repeat(m[np.newaxis,:,14:50], logft.shape[0],axis=0)
        bb = []
        for c in range(d.shape[0]):
            bb.append( pca[c].transform(logft[:,c,:])[:,:] )
        # lowpass
        fbb = filtfilt(b,a, np.array(bb), axis=1)[:,-450:-60,0]
        erbb.append(fbb.T)
    return np.array(erbb)


def riemann_fitERP(X, y):
    erpcov = ERPCovariances([1,2], estimator='oas')
    erpcov.fit(X, y)

    covar = erpcov.transform(X)

    tang = TangentSpace()
    tang.fit(covar)

    return erpcov, tang

def riemann_fitXdawn(X, y):
    erpcov = XdawnCovariances(nfilter=4, estimator='oas')
    erpcov.fit(X, y)

    covar = erpcov.transform(X)

    tang = TangentSpace()
    tang.fit(covar)

    return erpcov, tang

def riemann_transform(X, erpcov, tang):
    covar = erpcov.transform(X)
    proj = tang.transform(covar)

    return proj



class MyClassifier():
    def __init__(self, time=True, erbb=False, riemannERP=True, riemannXdawn=False, norm=True, featsel=False, bands=False):
        self.time = time
        self.erbb = erbb
        self.riemannERP = riemannERP
        self.riemannXdawn = riemannXdawn
        self.norm = norm
        self.featsel = featsel
        self.bands = bands

    def fit(self, X, y, pen=1.0):
        ''' 
            X -> (nepochs, n_times, n_channels) 
        '''
        nepochs = X.shape[0]

        X = preprocess(X) 

        X = X.transpose((0,2,1))

        if self.time or self.erbb or self.bands:
            if self.time:
                filtdata = filter(X.transpose((0,2,1)))
            if self.bands:
                bands = get_bands(X)
            if self.erbb:
                self.pca, self.m = get_spc(X)
                erbb = get_erbb_mor(X, self.pca, self.m)
            if self.time and self.erbb and self.bands:
                Xtr = np.concatenate([filtdata, erbb, bands], axis=2) 
            elif self.time and self.erbb:
                Xtr = np.concatenate([filtdata, erbb], axis=2) 
            elif self.time and self.bands:
                Xtr = np.concatenate([filtdata, bands], axis=2) 
            elif self.time:
                Xtr = filtdata
            else:
                Xtr = erbb

            Xtr = Xtr[:,::30,:].reshape(nepochs,-1)

        if self.riemannERP or self.riemannXdawn:
            if self.riemannERP:
                self.erpcov, self.tang = riemann_fitERP(X[...,-350:],y)
            elif self.riemannXdawn:
                self.erpcov, self.tang = riemann_fitXdawn(X[...,-350:],y)
            rproj = riemann_transform(X[...,-350:], self.erpcov, self.tang)
            if 'Xtr' in locals():
                Xtr = np.concatenate([Xtr, rproj], axis=1) 
            else:
                Xtr = rproj

        if self.norm:
            self.mean = Xtr.mean(axis=0)
            self.std = Xtr.std(axis=0)
            Xtr -= self.mean
            Xtr /= self.std

        if self.featsel:
            self.selfits = SelectKBest(f_classif, k=120).fit(Xtr, y)
            Xtr = self.selfits.transform(Xtr)
        #Xtr = Xtr.sum(axis=2)


        self.lr = LogisticRegression(penalty='l1', C=pen)
        self.lr.fit(Xtr, y)

    def transform(self, X):
        ''' 
            X -> (nepochs, n_times, n_channels) 
        '''

        X = X.copy()
        nepochs = X.shape[0]

        X = preprocess(X) 
        filtdata = filter(X)


        X = X.transpose((0,2,1))


        if self.time or self.erbb or self.bands:
            if self.time:
                filtdata = filter(X.transpose((0,2,1)))
            if self.bands:
                bands = get_bands(X)
            if self.erbb:
                erbb = get_erbb_mor(X, self.pca, self.m)
            if self.time and self.erbb and self.bands:
                Xtr = np.concatenate([filtdata, erbb, bands], axis=2) 
            elif self.time and self.erbb:
                Xtr = np.concatenate([filtdata, erbb], axis=2) 
            elif self.time and self.bands:
                Xtr = np.concatenate([filtdata, bands], axis=2) 
            elif self.time:
                Xtr = filtdata
            else:
                Xtr = erbb

            Xtr = Xtr[:,::30,:].reshape(nepochs,-1)

        if self.riemannERP or self.riemannXdawn:
            rproj = riemann_transform(X[...,-350:], self.erpcov, self.tang)
            if 'Xtr' in locals():
                Xtr = np.concatenate([Xtr, rproj], axis=1) 
            else:
                Xtr = rproj


        if self.norm:
            Xtr -= self.mean
            Xtr /= self.std

        if hasattr(self, 'featsel') and self.featsel:
            Xtr = self.selfits.transform(Xtr)

        return Xtr

    def predict(self, X):
        return self.lr.predict(self.transform(X))

    def predict_proba(self, X):
        return self.lr.predict_proba(self.transform(X))

