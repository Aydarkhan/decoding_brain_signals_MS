# Solution for Decoding Brain Signals, Cortana competition
The 4th place solution for the Cortana competition organized on the Microsoft Azure ML platform.

Despite its good performance in the competition, this was a highly exploratory work which lacks some structure and good coding style.

## Data
ECoG time series of 800 ms windows centered at the visual stimulus presentation: pictures of houses or faces. The task is to identify the category of presented picture from brain signals.

## In brief
The general pipeline is the following:
- learn a bunch of logistic regression classifiers with different combination of features
- select the best models based on 5-fold cross-validation 
- compute a weighted average of first-layer model predictions or stack them with SVM

Features:
- Event-related potential (ERP)
- Event-related broadband (ERBB)
- Band powers for 4-10 Hz estimated with wavelet transform
- Covariance matrix projected on Riemannian tangent space

## How to run prediction on Azure ML
*Warning* Training may take up to 1 hour.

After training the models with *train.py*, they will be saved in the *bundle* folder:
- Zip the *bundle* folder and upload it to Azure platform as a dataset.
- Create an experiment with the structure shown below.
- Copy-paste one of the classifying scripts into the 'Execute Python Script' module.

![AzureML predictive experiment](AzureML.png)
