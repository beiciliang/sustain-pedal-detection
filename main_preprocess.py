from __future__ import print_function  # (at top of module)
from __future__ import division
import os
import sys
import argparse
import pandas as pd
import numpy as np
import pretty_midi
import librosa

from global_config import *

def write_to_csv(rows, column_names, csv_fname):
    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(os.path.join(DIR_PEDAL_METADATA, csv_fname))
    
def prep_pedal_onset():
    """
    Get 500ms excerpts with/without pedal onset at 200ms    
    where pedal onset is obtained from midi file.
    """
    print('Start creating pedal-onset-dataset...')
    filename_segs = []
    filepaths = []
    ys = []
    categories = []
    for folder in FOLDERS:
        print('{}..'.format(folder))
        txt_path = os.path.join(DIR_PEDAL_METADATA,'{}.txt'.format(folder))
        filenames = np.genfromtxt(txt_path, dtype=None)

        pfolder_path = os.path.join(DIR_PEDAL_ONSET, folder, 'pedal-onset/')
        npfolder_path = os.path.join(DIR_PEDAL_ONSET, folder, 'non-pedal-onset/')
        if not os.path.exists(pfolder_path):
            os.makedirs(pfolder_path)
        if not os.path.exists(npfolder_path):
            os.makedirs(npfolder_path)

        for filename in filenames:
            print('  {}..'.format(filename))
            midi_path = os.path.join(DIR_RENDERED, '{}.mid'.format(filename))
            paudio_path = os.path.join(DIR_RENDERED, '{}-p.wav'.format(filename))
            npaudio_path = os.path.join(DIR_RENDERED, '{}-np.wav'.format(filename))
            paudio, sr = librosa.load(paudio_path, sr=SR)
            npaudio, sr = librosa.load(npaudio_path, sr=SR)

            # get ground truth pedal onset time from midi
            pm = pretty_midi.PrettyMIDI(midi_path)
            pedal_v = []
            pedal_t = []
            for control_change in pm.instruments[0].control_changes:
                if control_change.number == 64:
                    pedal_v.append(control_change.value)
                    pedal_t.append(control_change.time)

            pedal_onset = []
            for i,v in enumerate(pedal_v):
                if i>0 and v>=64 and pedal_v[i-1]<64:
                    pedal_onset.append(pedal_t[i])   

            pedal_onset_sp = librosa.time_to_samples(pedal_onset, sr=SR)

            for seg_idx, sp in enumerate(pedal_onset_sp):
                start_sp = int(sp - TRIM_SECOND_BEFORE * SR)
                end_sp = int(sp + TRIM_SECOND_AFTER * SR)
                newfilename = filename.replace('/','-')

                if start_sp > 0 and end_sp < len(npaudio):
                    pout_name = '{}-p_{}.wav'.format(newfilename, seg_idx)
                    pout_path = os.path.join(pfolder_path, pout_name)            
                    librosa.output.write_wav(pout_path, paudio[start_sp:end_sp], SR)
                    filename_segs.append(pout_name.rstrip('.wav'))
                    filepaths.append(os.path.join(folder, 'pedal-onset/', pout_name))
                    ys.append(1)
                    categories.append(folder)

                    npout_name = '{}-np_{}.wav'.format(newfilename, seg_idx)
                    npout_path = os.path.join(npfolder_path, npout_name)
                    librosa.output.write_wav(npout_path, npaudio[start_sp:end_sp], SR)
                    filename_segs.append(npout_name.rstrip('.wav'))
                    filepaths.append(os.path.join(folder, 'non-pedal-onset/', npout_name))
                    ys.append(0)
                    categories.append(folder)

    write_to_csv(zip(*[filename_segs, filepaths, ys, categories]), 
                 ['filename', 'filepath', 'label', 'category'], 
                 'pedal-onset_vd.csv')
    print('pedal-onset_vd.csv is saved!')
    

def prep_pedal_segment():
    """
    Get varient length excerpts with/without pedal effect    
    where the length is decided by midi file.
    """
    print('Start creating pedal-segment-dataset...')
    filename_segs = []
    filepaths = []
    ys = []
    categories = []
    min_sp = int(MIN_SRC * SR)
    max_sp = int(MAX_SRC * SR)
    for folder in FOLDERS:
        print('{}..'.format(folder))
        txt_path = os.path.join(DIR_PEDAL_METADATA,'{}.txt'.format(folder))
        filenames = np.genfromtxt(txt_path, dtype=None)

        pfolder_path = os.path.join(DIR_PEDAL_SEGMENT, folder, 'pedal-segment/')
        npfolder_path = os.path.join(DIR_PEDAL_SEGMENT, folder, 'non-pedal-segment/')
        if not os.path.exists(pfolder_path):
            os.makedirs(pfolder_path)
        if not os.path.exists(npfolder_path):
            os.makedirs(npfolder_path)

        for filename in filenames:
            print('  {}..'.format(filename))
            # get pedal segment from midi
            midi_path = os.path.join(PATH_DATASET, '{}.mid'.format(filename))
            pm = pretty_midi.PrettyMIDI(midi_path)
            pedal_v = []
            pedal_t = []
            for control_change in pm.instruments[0].control_changes:
                if control_change.number == 64:
                    pedal_v.append(control_change.value)
                    pedal_t.append(control_change.time)

            pedal_onset = []
            pedal_offset = []
            for i,v in enumerate(pedal_v):
                if i>0 and v>=64 and pedal_v[i-1]<64:
                    pedal_onset.append(pedal_t[i])
                elif i>0 and v<64 and pedal_v[i-1]>=64:
                    pedal_offset.append(pedal_t[i])

            pedal_offset = [t for t in pedal_offset if t > pedal_onset[0]]
            seg_idxs = np.min([len(pedal_onset), len(pedal_offset)])
            pedal_offset = pedal_offset[:seg_idxs]
            pedal_onset = pedal_onset[:seg_idxs]
            for seg_idx, offset in enumerate(pedal_offset):
                if offset != pedal_offset[-1] and offset > pedal_onset[seg_idx] and offset < pedal_onset[seg_idx+1]:
                    correct_pedal_data = True
                elif offset == pedal_offset[-1] and offset > pedal_onset[seg_idx]:
                    correct_pedal_data = True
                else:
                    correct_pedal_data = False

            if correct_pedal_data:
                pedal_onset_sp = librosa.time_to_samples(pedal_onset, sr=SR)
                pedal_offset_sp = librosa.time_to_samples(pedal_offset, sr=SR)
                paudio_path = os.path.join(DIR_RENDERED, '{}-p.wav'.format(filename))
                npaudio_path = os.path.join(DIR_RENDERED, '{}-np.wav'.format(filename))
                paudio, sr = librosa.load(paudio_path, sr=SR)
                npaudio, sr = librosa.load(npaudio_path, sr=SR)
                for seg_idx, start_sp in enumerate(pedal_onset_sp):
                    end_sp = pedal_offset_sp[seg_idx]
                    len_sp = end_sp - start_sp
                    if len_sp > max_sp:
                        end_sp = start_sp + max_sp

                    if len_sp >= min_sp and end_sp < len(npaudio):
                        newfilename = filename.replace('/','-')
                        pout_name = '{}-p_{}.wav'.format(newfilename, seg_idx)
                        pout_path = os.path.join(pfolder_path, pout_name)            
                        librosa.output.write_wav(pout_path, paudio[start_sp:end_sp], SR)
                        filename_segs.append(pout_name.rstrip('.wav'))
                        filepaths.append(os.path.join(folder, 'pedal-segment/', pout_name))
                        ys.append(1)
                        categories.append(folder)

                        npout_name = '{}-np_{}.wav'.format(newfilename, seg_idx)
                        npout_path = os.path.join(npfolder_path, npout_name)
                        librosa.output.write_wav(npout_path, npaudio[start_sp:end_sp], SR)
                        filename_segs.append(npout_name.rstrip('.wav'))
                        filepaths.append(os.path.join(folder, 'non-pedal-segment/', npout_name))
                        ys.append(0)
                        categories.append(folder)

    write_to_csv(zip(*[filename_segs, filepaths, ys, categories]), 
                 ['filename', 'filepath', 'label', 'category'], 
                 'pedal-segment_vd.csv')
    print('pedal-segment_vd.csv is saved!')    
    


def print_usage():
    print('This script trims excerpts from origninal audio files and saves them as new wav files.')
    print('$ python main_preprocess.py $dataset_name$')
    print('Example:')
    print('$ python main_preprocess.py pedal-onset-dataset')
    print('$ python main_preprocess.py pedal-segment-dataset')
    print('')
    print('Ps. Make sure you have the rendered dataset already and set the dirs/paths in config.json')
    
    
def main(args):
    dataset_name = args.dataset_name
    if dataset_name == 'pedal-onset-dataset':
        prep_pedal_onset()
    elif dataset_name == 'pedal-segment-dataset':
        prep_pedal_segment()
    else:
        print_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Preprocess the rendered audio to trim it into excerpts.") 
    parser.add_argument("dataset_name", type=str, help="name of the dataset.")
    main(parser.parse_args()) 