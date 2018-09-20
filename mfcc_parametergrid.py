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
config.gpu_options.per_process_gpu_memory_fraction = 0.3 # percentage to be used
set_session(tf.Session(config=config))

from kapre.time_frequency import Melspectrogram
from global_config import *

from multiprocessing import Pool
N_JOBS = 50

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid
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

gps = [{"C": [0.1, 2.0, 8.0, 32.0], "kernel": ['rbf'],
        "gamma": [0.5 ** i for i in [3, 5, 7, 9, 11, 13]] + ['auto']},
       {"C": [0.1, 2.0, 8.0, 32.0], "kernel": ['linear']}
      ]
classifier = SVC

filenames = ['onset', 'segment']
dataroots = [DIR_PEDAL_ONSET_NPY, DIR_PEDAL_SEGMENT_NPY]
    
for filename in filenames:
    
    loss_regs = []
    acc_regs = []
    auc_regs = []
    print('===== Pedal-{} SVC Performance ====='.format(filename))
    x_train, y_train = load_xy(filename, task_name='train')
    x_valid, y_valid = load_xy(filename, task_name='valid')
        
    for parameter in list(ParameterGrid(gps)):
        if parameter['kernel']=='linear':
            clf = SVC(kernel='linear', C=parameter['C']).fit(x_train, y_train)
        else:
            clf = SVC(kernel=parameter['kernel'], C=parameter['C'], gamma=parameter['gamma']).fit(x_train, y_train)

        y_pred = clf.predict(x_valid)
        loss_reg = log_loss(y_valid, y_pred)
        acc_reg = clf.score(x_valid, y_valid)
        auc_reg = roc_auc_score(y_valid, y_pred)

        print('{}'.format(parameter))
        print("      valid set loss: {}".format(loss_reg))
        print("  valid set accuracy: {}".format(acc_reg))
        print("       valid set auc: {}".format(auc_reg))
        
        loss_regs.append(loss_reg)
        acc_regs.append(acc_reg)
        auc_regs.append(auc_reg)
        
    save_npz_path = os.path.join(DIR_SAVE_MODEL,'small-{}_mfcc_svc_performance.npz'.format(filename))
    np.savez(save_npz_path, parameter=list(ParameterGrid(gps)), loss=loss_regs, acc=acc_regs, auc=auc_regs)