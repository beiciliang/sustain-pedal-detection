from __future__ import print_function
from __future__ import division
import os, sys
import numpy as np
import pandas as pd
from builtins import range
from sklearn.metrics import roc_auc_score
import librosa, librosa.display
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Flatten, Input, Reshape, Dropout, Permute
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"]="0" # the number of the GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # percentage to be used
set_session(tf.Session(config=config))

from kapre.time_frequency import Melspectrogram
from global_config import *

from multiprocessing import Pool
N_JOBS = 50

import multiprocessing
import logging
import cPickle as cP
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, roc_auc_score

class OptionalStandardScaler(StandardScaler):
    def __init__(self, on=False):
        self.on = on  # bool
        if self.on:
            super(OptionalStandardScaler, self).__init__(with_mean=True, with_std=True)
        else:
            super(OptionalStandardScaler, self).__init__(with_mean=False, with_std=False)

def load_xy(filename, task_name):
    
    npy_filename = 'small-{}_{}_mfcc.npy'.format(filename, task_name)
    x = np.load(os.path.join(DIR_SAVE_MODEL, npy_filename))
    
    csv_filename = 'pedal-{}_npydf_small.csv'.format(filename)
    df = pd.read_csv(os.path.join(DIR_PEDAL_METADATA, csv_filename))
    task_data = df.loc[df['category'] == task_name]   
    y = task_data.label.values
    return x, y
    
n_cpu = multiprocessing.cpu_count()
n_jobs = int(n_cpu * 0.8)
print('There are {} cpu available, {} (80%) of them will be used for our jobs.'.format(n_cpu, n_jobs))

gps = [{"C": [0.1, 2.0, 8.0, 32.0], "kernel": ['rbf'],
        "gamma": [0.5 ** i for i in [3, 5, 7, 9, 11, 13]] + ['auto']},
       {"C": [0.1, 2.0, 8.0, 32.0], "kernel": ['linear']}
      ]
classifier = SVC
dataroots = [DIR_PEDAL_ONSET_NPY, DIR_PEDAL_SEGMENT_NPY]
filenames = ['onset', 'segment']

for filename in filenames:
        
    x_train, y_train = load_xy(filename, task_name='train')
    x_valid, y_valid = load_xy(filename, task_name='valid')
    x = np.concatenate((x_train, x_valid), axis=0)
    y = np.concatenate((y_train, y_valid), axis=0)
    cv = [(np.arange(len(x_train)), np.arange(len(x_train),len(x)))]
    clname = classifier.__name__
    estimators = [('stdd', OptionalStandardScaler()), ('clf', classifier())]
    pipe = Pipeline(estimators)

    params = []
    for dct in gps:
        sub_params = {'stdd__on': [True, False]}
        sub_params.update({'clf__' + key: value for (key, value) in dct.iteritems()})
        params.append(sub_params)

    clf = GridSearchCV(pipe, params, cv=cv, n_jobs=n_jobs, pre_dispatch='8*n_jobs').fit(x, y)
    save_npy_path = os.path.join(DIR_SAVE_MODEL,'small-{}_mfcc_svc_best_params.npy'.format(filename))
    np.save(save_npy_path, [clf.best_params_])
    print('best score of pedal-{} {}: {}'.format(filename, clname, clf.best_score_))
    print(clf.best_params_)    