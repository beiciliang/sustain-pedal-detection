# sustain-pedal-detection

Companion codes for the submissions:

Beici Liang, György Fazekas, Mark Sandler. "Piano Sustain-Pedal Detection Using Convolutional Neural Networks" published in ICASSP 2019 and "Transfer Learning for Piano Sustain-Pedal Detection" published in IJCNN 2019.

<img src="https://github.com/beiciliang/sustain-pedal-detection/blob/master/framework.png" width="500">

## Index

* `0. pedal midi info.ipynb`: understand MIDI files and how the ground-truth annotations are extracted

* `1.1 dataset preparation.ipynb` and `1.2 sub-dataset preparation.ipynb`: how to build the dataset and generate excerpts

#### Works related to the ICASSP paper

* `2.1 pedal onset classification.ipynb`: how to train `Conv2D-onset` and save the model with the highest AUC in `./save-model/onset_multi_kernel`.

* `2.2 pedal segment classification.ipynb`: how to train `Conv2D-segment` and save the model with the highest AUC in `./save-model/segment_multi_kernel`.

* `2.3 how mfcc performs on the small dataset.ipynb`: compare with SVM using MFCC features. Performance on detecting pedal onset and pedalled segment are saved in `./save-result/small-onset_mfcc_svc_performance.npz` and `./save-result/small-onset_mfcc_svc_performance.npz`, respectively.

* `3. piece-wise detection.ipynb`: how to fuse the decision outputs from `Conv2D-onset` and `Conv2D-segment` so as to perform the detection on a piano piece. Evaluation results are saved in `./save-result/psegment-testresult_onset98_seg98.csv`.

#### Works related to the IJCNN paper and Chapter 7 in Beici's PhD Thesis

* `4. effect of cnn settings.ipynb`: how different configurations affect the model performance. After this, best model for detecting pedal onset and pedalled segment are saved in `./save-model/sub-onset_cnnkernel-melspectrogram_l4c13` and `./save-model/sub-segment_cnnkernel-melspectrogram_multift`, respectively.

* `5.1 test on chopin audio data (retrain last layer).ipynb`: how to fine-tune the model trained using synthesised data to be used in the detection on real acoustic recordings.

* `5.2 test on chopin audio data.ipynb`: how to do the detection based on our proposed transfer learning strategy.

* `visualise layers.ipynb`: visualise what have been learned in the neural networks using deconvolution.

## Requirements

Codes are based on the following settings and their corresponding versions. 

Setting | Version
------------ | -------------
OS | Centos 7.3
GPU | Titan Xp
module | cuda/8.0-cudnn5.1
Python | 2.7.5

Python dependencies can be installed by
```
pip install -r requirements.txt
```

You need to install [Jupyter Notebook](http://jupyter.org/) to run `.ipynb` in your local browser.
