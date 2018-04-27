# Feature-Extraction-for-Speech-Recognition
Python Implementation of the Feature Extraction Process in Kaldi

## Brief Discription
This repository contains Python scripts for the feature extraction process in speech recognition systems. It is originated from Kaldi (https://github.com/kaldi-asr/kaldi), but may be more flexible for most speech recognition systems, especially end-to-end ones.

## Contents
1. feature_extraction_template.py: a template for the feature extraction process, blocks which need to be filled by the user are commented with the keyword "BLOCK"
2. utils/extract_window.py: the preprocessing and window slicing functions
3. utils/fft2melmx.py: an adopted script calculating mel weights for the conversion from the fft feature to the mel feature, see the comments in the file for details
4. utils/deltas.py: the delta feature calculation function used in TensorFlow graphs
5. utils/deltas_np.py: the delta feature calculation function which can be used without TensorFlow

## Arguments
Most of the arguments can be modified in feature_extraction_template.py. The default arguments in the template are used for the wide residual BLSTM network ([WRBN](https://github.com/jheymann85/chime4_backend)) based acoustic model.

## Known Limitations and Comments
1. The feature_extraction_template.py only shows the feature extraction process with the usage of TensorFlow. It should be easy to extend it to the version without TensorFlow (using utils/deltas_np.py).
2. Currently the *log energy pre window* function in Kaldi is not supported. This function is seldomly used in most of my work though. Contributions are welcome.
3. Currently we only support filter-bank (fbank) features. It should be easy to extend to other popular features such as MFCC.
