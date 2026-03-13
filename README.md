# Audio Signal Classification - Spectral Feature Extraction

Audio classification pipeline that extracts time-frequency features from raw audio and trains a TensorFlow classifier with stratified cross-validation.

## Features

- MFCC extraction
- Spectral centroid extraction
- Zero-crossing rate extraction
- FFT magnitude preprocessing
- Dense neural network classifier in TensorFlow
- Stratified K-Fold evaluation with mean and std accuracy reporting
