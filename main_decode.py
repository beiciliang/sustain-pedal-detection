from __future__ import print_function  # (at top of module)
from __future__ import division
import os
import sys
import argparse
import pandas as pd
import librosa
import numpy as np
import multiprocessing

from global_config import *

def load_save_pedal_onset_npy(track_id):
    """
    Load, decode, and save tracks of pedal onset dataset.
    Load/Save paths are set by `config.json`.
    track_id : integer. e.g. 2
    """
    audio_path = os.path.join(DIR_PEDAL_ONSET,DF_ONSET.loc[[track_id]].filepath.values[0])
    src, _ = librosa.load(audio_path, sr=SR)
    src = src.astype(np.float16)    
    npy_path = os.path.join(DIR_PEDAL_ONSET_NPY,DF_ONSET.loc[[track_id]].filepath.values[0].split('.')[0]+'.npy')
    np.save(npy_path, src)
    
    print('  {}'.format(DF_ONSET.loc[[track_id]].filename.values[0]))


def decode_pedal_onset():
    """
    Decode rendered onset dataset and store them in numpy arrays.
    16-bit Float, SR=44100, and 0.5s
    """
    for folder in FOLDERS:
        print('{}..'.format(folder))

        pfolder_path = os.path.join(DIR_PEDAL_ONSET_NPY, folder, 'pedal-onset/')
        npfolder_path = os.path.join(DIR_PEDAL_ONSET_NPY, folder, 'non-pedal-onset/')
        if not os.path.exists(pfolder_path):
            os.makedirs(pfolder_path)
        if not os.path.exists(npfolder_path):
            os.makedirs(npfolder_path)
            
        tracks_folder = DF_ONSET['category'] == folder
        indices = DF_ONSET.loc[tracks_folder].index

        # decoding
        p = multiprocessing.Pool()
        p.map(load_save_pedal_onset_npy, indices)
        
        
def load_save_pedal_segment_npy(track_id):
    """
    Load, decode, and save tracks of pedal segment dataset.
    Load/Save paths are set by `config.json`.
    track_id : integer. e.g. 2
    """
    audio_path = os.path.join(DIR_PEDAL_SEGMENT,DF_SEGMENT.loc[[track_id]].filepath.values[0])
    src, _ = librosa.load(audio_path, sr=SR, duration=LEN_SRC)
    if len(src) < NSP_SRC:
        tile_times = int(np.ceil(NSP_SRC/len(src)))
        src = np.tile(src, tile_times)[:NSP_SRC]
    else:
        src = src[:NSP_SRC]
    src = src.astype(np.float16)    
    npy_path = os.path.join(DIR_PEDAL_SEGMENT_NPY,DF_SEGMENT.loc[[track_id]].filepath.values[0].split('.')[0]+'.npy')
    np.save(npy_path, src)
    
    print('  {}'.format(DF_SEGMENT.loc[[track_id]].filename.values[0]))


def decode_pedal_segment():
    """
    Decode rendered segment dataset and store them in numpy arrays.
    16-bit Float, SR=44100, and 2s
    """    
    for folder in FOLDERS:
        print('{}..'.format(folder))

        pfolder_path = os.path.join(DIR_PEDAL_SEGMENT_NPY, folder, 'pedal-segment/')
        npfolder_path = os.path.join(DIR_PEDAL_SEGMENT_NPY, folder, 'non-pedal-segment/')
        if not os.path.exists(pfolder_path):
            os.makedirs(pfolder_path)
        if not os.path.exists(npfolder_path):
            os.makedirs(npfolder_path)
            
        tracks_folder = DF_SEGMENT['category'] == folder
        indices = DF_SEGMENT.loc[tracks_folder].index

        # decoding
        p = multiprocessing.Pool()
        p.map(load_save_pedal_segment_npy, indices)
        
        
def print_usage():
    print('This script decode audio excerpts and saves them as npy files.')
    print('$ python main_decode.py $dataset_name$')
    print('Example:')
    print('$ python main_decode.py pedal-onset-dataset')
    print('$ python main_decode.py pedal-segment-dataset')
    print('')
    print('Ps. Make sure you have run the preprocess and set the dirs/paths in config.json')
    
    
def main(args):
    dataset_name = args.dataset_name
    if dataset_name == 'pedal-onset-dataset':
        vd_pedal_onset = os.path.join(DIR_PEDAL_METADATA, 'pedal-onset_vd.csv')
        global DF_ONSET
        DF_ONSET = pd.read_csv(vd_pedal_onset)
        decode_pedal_onset()
    elif dataset_name == 'pedal-segment-dataset':
        vd_pedal_segment = os.path.join(DIR_PEDAL_METADATA, 'pedal-segment_vd.csv')
        global DF_SEGMENT
        DF_SEGMENT = pd.read_csv(vd_pedal_segment)
        decode_pedal_segment()
    else:
        print_usage()
 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Decode audio excerpts and saves them as npy files.") 
    parser.add_argument("dataset_name", type=str, help="name of the dataset.")
    main(parser.parse_args()) 