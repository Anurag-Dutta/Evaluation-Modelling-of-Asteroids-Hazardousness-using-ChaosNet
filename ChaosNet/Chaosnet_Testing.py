"""
Created on Wed Dec 28, 2022

Author: Anurag Dutta (anuragdutta.research@gmail.com)
Code Description: A python code to test the efficacy of stand-alone ChaosNet on the ASTEROID dataset.

"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import ChaosFEX.feature_extractor as CFX
from Codes import chaosnet


'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________
    1 (Fraud)          -     1
    0 (Not Fraud)      -     0
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________
    nasa            -   Complete ASTEROID dataset.    
    X               -   Data attributes.
    y               -   Corresponding labels for X.
    X_train         -   Data attributes for training (80% of the dataset).
    y_train         -   Corresponding labels for X_train.
    X_test          -   Data attributes for testing (20% of the dataset).
    y_test          -   Corresponding labels for X_test.
    X_train_norm    -   Normalized training data attributes (X_train).
    X_test_norm     -   Normalized testing data attributes (X_test).
_______________________________________________________________________________

CFX hyperparameter description:
_______________________________________________________________________________
    INITIAL_NEURAL_ACTIVITY         -   Initial Neural Activity.
    EPSILON                         -   Noise Intensity.
    DISCRIMINATION_THRESHOLD        -   Discrimination Threshold.
    
    Source: Harikrishnan N.B., Nithin Nagaraj,
    When Noise meets Chaos: Stochastic Resonance in Neurochaos Learning,
    Neural Networks, Volume 143, 2021, Pages 425-435, ISSN 0893-6080,
    https://doi.org/10.1016/j.neunet.2021.06.025.
    (https://www.sciencedirect.com/science/article/pii/S0893608021002574)
_______________________________________________________________________________


'''



#import the ASTEROID Dataset
nasa = np.array(pd.read_csv('nasa.txt', sep="," ,header=None))


#reading data and labels from the dataset
X, y = nasa[:,range(0,nasa.shape[1]-1)], nasa[:,nasa.shape[1]-1]
y = y.reshape(len(y),1).astype(str)
y = np.char.replace(y, '0.0', '0', count=None)
y = np.char.replace(y, '1.0', '1', count=None)
y = y.astype(int)



#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


#Normalisation - Column-wise
X_train_norm = (X_train - np.min(X_train,0)) / (np.max(X_train,0) - np.min(X_train,0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = (X_test - np.min(X_test,0)) / (np.max(X_test,0) - np.min(X_test,0))
X_test_norm = X_test_norm.astype(float)


#Testing
PATH = os.getcwd()
RESULT_PATH = PATH + '/CFX TUNING/RESULTS/'

INA = np.load(RESULT_PATH+"/h_Q.npy")[0]
EPSILON_1 = np.load(RESULT_PATH+"/h_EPS.npy")[0]
DT = np.load(RESULT_PATH+"/h_B.npy")[0]

FEATURE_MATRIX_TRAIN = CFX.transform(X_train_norm, INA, 10000, EPSILON_1, DT)
FEATURE_MATRIX_VAL = CFX.transform(X_test_norm, INA, 10000, EPSILON_1, DT)

mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, y_train, FEATURE_MATRIX_VAL)

f1 = f1_score(y_test, Y_PRED, average='macro')
print('TESTING F1 SCORE', f1)

np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
