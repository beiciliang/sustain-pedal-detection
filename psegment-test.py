from __future__ import print_function
from __future__ import division
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from builtins import range
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import mir_eval
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

reg_w = 1e-4
batch_size = 1

def model_multi_kernel_shape(n_out, input_shape, out_activation='softmax'):
    """

    Symbolic summary:
    > c2' - p2 - c2 - p2 - c2 - p2 - c2 - p3 - d1
    where c2' -> multiple kernel shapes

    Parameters
    ----------
        n_out: integer, number of output nodes
        input_shape: tuple, an input shape, which doesn't include batch-axis.
        out_activation: activation function on the output
    """
    audio_input = Input(shape=input_shape)

    x = Melspectrogram(n_dft=N_FFT, n_hop=HOP_LENGTH, sr=SR, n_mels=128, power_melgram=2.0, return_decibel_melgram=True)(audio_input)
    x = BatchNormalization(axis=channel_axis)(x)

    x1 = Conv2D(7, (20, 3), padding='same', kernel_regularizer=keras.regularizers.l2(reg_w))(x)
    x2 = Conv2D(7, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(reg_w))(x)
    x3 = Conv2D(7, (3, 20), padding='same', kernel_regularizer=keras.regularizers.l2(reg_w))(x)

    x = Concatenate(axis=channel_axis)([x1, x2, x3])

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(21, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(reg_w))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(21, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(reg_w))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(21, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(reg_w))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((4, 4), padding='same')(x)
    x = Dropout(0.25)(x)

    x = GlobalAveragePooling2D()(x)

    out = Dense(n_out, activation=out_activation, kernel_regularizer=keras.regularizers.l2(reg_w))(x)

    model = Model(audio_input, out)

    return model

def data_gen(audio_data, n_detect, nsp_excerpt, type_excerpt, hop_length=HOP_LENGTH):
    """Data generator.
    excerpt: data of one audio file.
    n_detect: number of excerpts to be detected.
    nsp_detect: number of samples in one excerpt.
    """
    
    tile_times = int(np.ceil(NSP_SRC/nsp_excerpt))

    while True:
        for i in range(n_detect):
            
            if type_excerpt == 'onset':
                src_batch = np.array([audio_data[int(i*hop_length):int(i*hop_length+nsp_excerpt)]], dtype=K.floatx())
            elif type_excerpt == 'segment':
                src_batch = np.array([np.tile(audio_data[int(i*hop_length):int(i*hop_length+nsp_excerpt)],tile_times)[:NSP_SRC]],
                                     dtype=K.floatx())
                
            src_batch = src_batch[:, np.newaxis, :]  # make (batch, N) to (batch, 1, N) for kapre compatible
            
            yield src_batch
            

def intervals1tointervals01(segintervals1, paudio_duration):
    idx2del = []
    for idx in np.arange(1,len(segintervals1)):
        if segintervals1[idx-1][1] >= segintervals1[idx][0]:
            segintervals1[idx] = [segintervals1[idx-1][0],segintervals1[idx][1]]
            idx2del.append(idx-1)           
    segintervals1 = np.delete(segintervals1, idx2del, axis=0)  
    
    labels = []
    segintervals01 = np.zeros((len(segintervals1)*2+1,2))
    
    for idx in range(len(segintervals01)):
        if idx==0:
            segintervals01[idx] = [0, segintervals1[0][0]]
            labels.append('np')
        elif idx==len(segintervals01)-1:
            segintervals01[idx] = [segintervals1[-1][-1],paudio_duration]
            labels.append('np')
        elif idx%2:
            segintervals01[idx] = segintervals1[int(np.floor(idx/2))]
            labels.append('p')
        else:
            segintervals01[idx] = [segintervals1[int(np.floor(idx/2)-1)][-1],segintervals1[int(np.floor(idx/2))][0]]
            labels.append('np')
            
    idx2del = []
    for idx, seginterval in enumerate(segintervals01):
        if seginterval[0]==seginterval[1]:
            idx2del.append(idx)
    segintervals01 = np.delete(segintervals01, idx2del, axis=0)
    labels = np.delete(labels, idx2del)
    
    return segintervals1, segintervals01, labels


def main(args):
    music_type = args.music_type
    onset_threshold = args.onset_threshold
    segment_threshold = args.segment_threshold

    if music_type == 'synth':
        dataset_name = 'pedal-times_test.npz'
        npz_path = os.path.join(DIR_PEDAL_METADATA, dataset_name)
    elif music_type == 'real':
        npz_dir = os.path.join(DIR_REAL_DATA, 'reference') 
        dataset_name = 'pedal-times_realaudio.npz'
        npz_path = os.path.join(npz_dir, dataset_name)
    else:
        print("Error: Please set the music_type to either synth or real!")

    tracks = np.load(npz_path)
    filenames = tracks['filename']
    pedal_offset_gt_tracks = tracks['pedal_offset']
    pedal_onset_gt_tracks = tracks['pedal_onset']

    # get model
    model_name = 'multi_kernel'
    segment_exp_name = 'segment_{}'.format(model_name)
    onset_exp_name = 'onset_{}'.format(model_name)

    onset_model = model_multi_kernel_shape(n_out=2,input_shape=ONSET_INPUT_SHAPE)
    onset_model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    segment_model = model_multi_kernel_shape(n_out=2,input_shape=SEGMENT_INPUT_SHAPE)
    segment_model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    # load weights
    onset_model.load_weights(os.path.join(DIR_SAVE_MODEL,"{}_best_weights.h5".format(onset_exp_name)))
    segment_model.load_weights(os.path.join(DIR_SAVE_MODEL,"{}_best_weights.h5".format(segment_exp_name)))

    # do detection
    filename_records = []
    accuracys = []
    precisions = []
    recalls = []
    fscores = []
    support0s = []
    support1s = []
    fp_rates = []
    fn_rates = []
    # append to lists
    filename_records = []
    support0s = []
    support1s = []
    acc01_frms = []
    p1_frms = []
    r1_frms = []
    f1_frms = []
    fp_rates = []
    fn_rates = []
    # boundary matrixs
    boundary_wins = []
    p1_sbrs = []
    r1_sbrs = []
    f1_sbrs = []
    r2e_deviation1s = []
    e2r_deviation1s = []
    p01_sbrs = []
    r01_sbrs = []
    f01_sbrs = []
    r2e_deviation01s = []
    e2r_deviation01s = []
    # structural matrixs
    p_pairwises = []
    r_pairwises = []
    f_pairwises = []
    nce_overs = []
    nce_unders = []
    nce_fs = []
    rand_indexs = []
    adjrand_indexs = []
    mutual_infos = []
    adjmutual_infos = []
    normmutual_infos = []

    for filename_idx, filename in enumerate(filenames):  
        pedal_offset_gt = np.array(pedal_offset_gt_tracks[filename_idx])
        pedal_onset_gt = np.array(pedal_onset_gt_tracks[filename_idx])
        if music_type == 'synth':
            paudio_path = os.path.join(DIR_RENDERED, '{}-p.wav'.format(filename))
        elif music_type == 'real':
            paudio_dir = os.path.join(DIR_REAL_DATA, '{}'.format(filename)) 
            paudio_path = os.path.join(paudio_dir, '{}.wav'.format(filename))

        paudio, sr = librosa.load(paudio_path, sr=SR) 
        print("{}...".format(filename))
        len_onset_shape = int(SR * (TRIM_SECOND_BEFORE + TRIM_SECOND_AFTER))
        onsethop_length = HOP_LENGTH
        onsethop_duration = onsethop_length/SR
        n_ponset = int(np.ceil((len(paudio)-len_onset_shape)/onsethop_length))
        gen_ponset = data_gen(paudio, n_ponset, len_onset_shape, 'onset', hop_length=onsethop_length)
        pred_ponset = onset_model.predict_generator(gen_ponset, n_ponset // batch_size)
        pred_ponset_filter = medfilt(pred_ponset[:,1],15)
        frmtime_ponset = np.arange(n_ponset)*onsethop_duration+TRIM_SECOND_BEFORE

        len_segment_shape = int(SR * MIN_SRC)
        seghop_length = HOP_LENGTH*10
        seghop_duration = seghop_length/SR
        n_psegment = int(np.ceil((len(paudio)-len_segment_shape)/seghop_length))
        gen_psegment = data_gen(paudio, n_psegment, len_segment_shape, 'segment', hop_length=seghop_length)
        pred_psegment = segment_model.predict_generator(gen_psegment, n_psegment // batch_size)
        pred_psegment_filter = medfilt(pred_psegment[:,1],3)
        frmtime_psegment = np.arange(n_psegment)*seghop_duration+MIN_SRC/2
        paudio_firstonsettime = librosa.frames_to_time(librosa.onset.onset_detect(y=paudio, sr=SR), sr=SR)[0]
        n_segment_tozero=0
        for t in frmtime_psegment:
            if t < paudio_firstonsettime:
                n_segment_tozero+=1
            else:
                break        
        pred_psegment_filter[:n_segment_tozero] = 0

        pred_ponset_todetect = np.copy(pred_ponset_filter)
        pred_ponset_todetect[pred_ponset_todetect<onset_threshold]=0
        pred_ponset_todetect[pred_ponset_todetect>=onset_threshold]=1

        pred_psegment_todetect = np.copy(pred_psegment_filter)
        pred_psegment_todetect[pred_psegment_todetect<segment_threshold]=0
        pred_psegment_todetect[pred_psegment_todetect>=segment_threshold]=1

        # decide the initial indexes of pedal segment boundary
        onseg_initidxs = []
        offseg_initidxs = []
        for idx, v in enumerate(pred_psegment_todetect):
            if idx>0 and idx<len(pred_psegment_todetect)-1:
                if pred_psegment_todetect[idx-1]==0 and v==1 and pred_psegment_todetect[idx+1]==1:
                    onseg_initidxs.append(idx-1)
                elif pred_psegment_todetect[idx-1]==1 and v==1 and pred_psegment_todetect[idx+1]==0:
                    offseg_initidxs.append(idx+1)

        if offseg_initidxs[0] <= onseg_initidxs[0]:
            del offseg_initidxs[0]
        if onseg_initidxs[-1] >= offseg_initidxs[-1]:
            del onseg_initidxs[-1]

        if (len(onseg_initidxs) != len(offseg_initidxs)) or not len(pedal_offset_gt) or not len(pedal_onset_gt):
            print(" skip!")
        else:
            onseg_idxs = []
            offseg_idxs = []
            for idx in range(len(onseg_initidxs)):
                if onseg_initidxs[idx] < offseg_initidxs[idx]:
                    onseg_idxs.append(onseg_initidxs[idx])
                    offseg_idxs.append(offseg_initidxs[idx])

            if not len(onseg_idxs) or not len(offseg_idxs):
                print("  no detection!")  

            else:
                # decide the boundary times in seconds, combining the effect of pedal onset
                onseg_times = []
                offseg_times = []
                for idx, onseg_idx in enumerate(onseg_idxs):
                    onponset_idx = onseg_idx*10-5
                    if any(pred_ponset_todetect[onponset_idx-5:onponset_idx+5]):
                        offseg_idx = offseg_idxs[idx]
                        offseg_times.append(frmtime_psegment[offseg_idx])
                        onseg_times.append(frmtime_psegment[onseg_idx])
                segintervals_est = np.stack((np.asarray(onseg_times),np.asarray(offseg_times)), axis=-1)

                # set the ground truth and estimation results frame by frame
                paudio_duration = librosa.get_duration(y=paudio, sr=SR)
                n_frames = int(np.ceil(paudio_duration/seghop_duration))
                segframes_gt = np.zeros(n_frames)
                segframes_est = np.zeros(n_frames)

                pedal_offset_gt = np.array(tracks['pedal_offset'][filename_idx])
                pedal_onset_gt = np.array(tracks['pedal_onset'][filename_idx])
                longpseg_idx = np.where((pedal_offset_gt-pedal_onset_gt)>seghop_duration)[0]
                longseg_onset_gt = pedal_onset_gt[longpseg_idx]
                longseg_offset_gt = pedal_offset_gt[longpseg_idx]
                segintervals_gt = np.stack((longseg_onset_gt,longseg_offset_gt), axis=-1)

                for idx, onset_t in enumerate(longseg_onset_gt):
                    offset_t = longseg_offset_gt[idx]
                    onset_frm = int(onset_t//seghop_duration)
                    offset_frm = int(offset_t//seghop_duration)
                    segframes_gt[onset_frm:offset_frm] = 1

                for idx, onset_t in enumerate(onseg_times):
                    offset_t = offseg_times[idx]
                    onset_frm = int(onset_t//seghop_duration)
                    offset_frm = int(offset_t//seghop_duration)
                    segframes_est[onset_frm:offset_frm] = 1 

                # set the ground truth and estimation results as interval format
                segintervals1_gt, segintervals01_gt, labels_gt = intervals1tointervals01(segintervals_gt, paudio_duration)
                segintervals1_est, segintervals01_est, labels_est = intervals1tointervals01(segintervals_est, paudio_duration)

                # Metrics for frame-wise label 'p'
                acc01_frm = accuracy_score(segframes_gt,segframes_est)
                p1_frm, r1_frm, f1_frm, support = precision_recall_fscore_support(segframes_gt,segframes_est)
                tn, fp, fn, tp = confusion_matrix(segframes_gt,segframes_est).ravel()
                fp_rate = fp/(fp+tn)
                fn_rate = fn/(fn+tp)

                # performance matrix based on boundary annotation of 'p'
                # window depends on duration of a beat
                onset_env = librosa.onset.onset_strength(paudio, sr=SR)
                tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=SR)[0]
                beat_insecond = 60/tempo
                p1_sbr,r1_sbr,f1_sbr = mir_eval.segment.detection(segintervals1_gt, segintervals1_est, window=beat_insecond)
                r2e_deviation1, e2r_deviation1 = mir_eval.segment.deviation(segintervals1_gt, segintervals1_est)
                # performance matrix based on boundary annotation of both 'p' and 'np' 
                p01_sbr,r01_sbr,f01_sbr = mir_eval.segment.detection(segintervals01_gt, segintervals01_est, window=beat_insecond)

                # performance matrix based on structural annotation
                scores = mir_eval.segment.evaluate(segintervals01_gt, labels_gt, segintervals01_est, labels_est)
                r2e_deviation01, e2r_deviation01 = [scores['Ref-to-est deviation'], scores['Est-to-ref deviation']]
                p_pairwise, r_pairwise, f_pairwise = [scores['Pairwise Precision'], scores['Pairwise Recall'], 
                                                      scores['Pairwise F-measure']]
                rand_index, adjrand_index = [scores['Rand Index'], scores['Adjusted Rand Index']]
                mutual_info, adjmutual_info, normmutual_info = [scores['Mutual Information'], scores['Adjusted Mutual Information'], 
                                                                scores['Normalized Mutual Information']]
                nce_over, nce_under, nce_f = [scores['NCE Over'], scores['NCE Under'], scores['NCE F-measure']]

                # append to lists
                filename_records.append(filename)
                support0s.append(support[0])
                support1s.append(support[1])
                acc01_frms.append(acc01_frm)
                p1_frms.append(p1_frm[1])
                r1_frms.append(r1_frm[1])
                f1_frms.append(f1_frm[1])
                fp_rates.append(fp_rate)
                fn_rates.append(fn_rate)
                # boundary matrixs
                boundary_wins.append(beat_insecond)
                p1_sbrs.append(p1_sbr)
                r1_sbrs.append(r1_sbr)
                f1_sbrs.append(f1_sbr)
                r2e_deviation1s.append(r2e_deviation1)
                e2r_deviation1s.append(e2r_deviation1)
                p01_sbrs.append(p01_sbr)
                r01_sbrs.append(r01_sbr)
                f01_sbrs.append(f01_sbr)
                r2e_deviation01s.append(r2e_deviation01)
                e2r_deviation01s.append(e2r_deviation01)
                # structural matrixs
                p_pairwises.append(p_pairwise)
                r_pairwises.append(r_pairwise)
                f_pairwises.append(f_pairwise)
                nce_overs.append(nce_over)
                nce_unders.append(nce_under)
                nce_fs.append(nce_f)
                rand_indexs.append(rand_index)
                adjrand_indexs.append(adjrand_index)
                mutual_infos.append(mutual_info)
                adjmutual_infos.append(adjmutual_info)
                normmutual_infos.append(normmutual_info)
                print("  done!")

    rows = zip(*[filename_records, support0s, support1s, acc01_frms, p1_frms, r1_frms, f1_frms, fp_rates, fn_rates, 
                 boundary_wins, p1_sbrs, r1_sbrs, f1_sbrs, r2e_deviation1s, e2r_deviation1s,
                 p01_sbrs, r01_sbrs, f01_sbrs, r2e_deviation01s, e2r_deviation01s,
                 p_pairwises, r_pairwises, f_pairwises, nce_overs, nce_unders, nce_fs, rand_indexs, 
                 adjrand_indexs, mutual_infos, adjmutual_infos, normmutual_infos])
    column_names =  ['filename_record', 'support0', 'support1', 'acc01_frm', 'p1_frm', 'r1_frm', 'f1_frm', 'fp_rate', 'fn_rate', 
                     'boundary_win', 'p1_sbr', 'r1_sbr', 'f1_sbr', 'r2e_deviation1', 'e2r_deviation1',
                     'p01_sbr', 'r01_sbr', 'f01_sbr', 'r2e_deviation01', 'e2r_deviation01',
                     'p_pairwise', 'r_pairwise', 'f_pairwise', 'nce_over', 'nce_under', 'nce_f', 'rand_index', 
                     'adjrand_index', 'mutual_info', 'adjmutual_info', 'normmutual_info']
    df = pd.DataFrame(rows, columns = column_names)

    if music_type == 'synth':
        df.to_csv('psegment-testresult_onset{}_seg{}.csv'.format(int(onset_threshold*100),int(segment_threshold*100)))
    elif music_type == 'real':
        df.to_csv('psegment-testresult-realaudio_onset{}_seg{}.csv'.format(int(onset_threshold*100),int(segment_threshold*100)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Batch testing the pedal detection method on music pieces.")
    parser.add_argument("music_type", 
                        type = str, 
                        help = "Test one either synth or real.")
    parser.add_argument("onset_threshold", 
                        type = float, 
                        default = 0.98, 
                        help = "Threshold on the output of Conv2D-onset")
    parser.add_argument("segment_threshold", 
                        type = float, 
                        default = 0.98, 
                        help = "Threshold on the output of Conv2D-segment")
    main(parser.parse_args())
