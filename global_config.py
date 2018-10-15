import json
from keras import backend as K

if K.image_data_format() == 'channels_first':
    print('Channel-first, i.e., (None, n_ch, n_freq, n_time)')
    channel_axis = 1
    freq_axis = 2
    time_axis = 3
else:
    print('Channel-last, i.e., (None, n_freq, n_time, n_ch)')
    channel_axis = 3
    freq_axis = 1
    time_axis = 2

# Constants
SR = 44100
N_FFT = 1024             # 23 ms
HOP_LENGTH = int(0.01*SR)  # 10 ms 
TRIM_SECOND_BEFORE = 0.2
TRIM_SECOND_AFTER = 0.3
ONSET_INPUT_SHAPE = (1, int(SR * (TRIM_SECOND_BEFORE + TRIM_SECOND_AFTER)))
MIN_SRC = 0.3
MAX_SRC = 2.3
LEN_SRC = 2.0
NSP_SRC = int(SR * LEN_SRC)
SEGMENT_INPUT_SHAPE = (1, NSP_SRC)
FOLDERS = ['train', 'valid', 'test']

# Paths
with open('config.json') as json_data:
    config = json.load(json_data)

DIR_RENDERED = config['dir_rendered']
DIR_PEDAL_METADATA = config['dir_pedal_metadata']
DIR_PEDAL_ONSET = config['dir_pedal_onset']
DIR_PEDAL_SEGMENT = config['dir_pedal_segment']
DIR_PEDAL_ONSET_NPY = config['dir_pedal_onset_npy']
DIR_PEDAL_SEGMENT_NPY = config['dir_pedal_segment_npy']
DIR_SAVE_MODEL = config['dir_save_model']
DIR_REAL_DATA = config['dir_real_data']