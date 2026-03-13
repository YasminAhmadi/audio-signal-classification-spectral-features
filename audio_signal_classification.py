import os
import numpy as np
import librosa
from scipy.fft import rfft
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

DATASET_DIR = "data/audio"
SR = 22050
DURATION = 3.0
N_MFCC = 20
FFT_BINS = 1024
N_SPLITS = 5
EPOCHS = 40
BATCH_SIZE = 32
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

def collect_files(root_dir):
    files, labels = [], []
    for label in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for name in os.listdir(class_dir):
            if name.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                files.append(os.path.join(class_dir, name))
                labels.append(label)
    return files, labels

def pad_or_trim(y, target_len):
    if len(y) >= target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)))

def extract_features(path):
    y, sr = librosa.load(path, sr=SR, mono=True, duration=DURATION)
    target_len = int(SR * DURATION)
    y = pad_or_trim(y, target_len)

    fft_mag = np.abs(rfft(y))
    if len(fft_mag) >= FFT_BINS:
        fft_feat = fft_mag[:FFT_BINS]
    else:
        fft_feat = np.pad(fft_mag, (0, FFT_BINS - len(fft_mag)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = centroid.mean(axis=1)
    centroid_std = centroid.std(axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean(axis=1)
    zcr_std = zcr.std(axis=1)

    return np.concatenate([mfcc_mean, mfcc_std, centroid_mean, centroid_std, zcr_mean, zcr_std, fft_feat])

def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    files, labels = collect_files(DATASET_DIR)
    if len(files) == 0:
        raise ValueError(f"No audio files found in {DATASET_DIR}")

    X = np.vstack([extract_features(f) for f in files])
    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]
        y_test_labels = y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = build_model(X_train.shape[1], num_classes)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0
        )

        probs = model.predict(X_test, verbose=0)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_test_labels, preds)
        fold_accuracies.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")

    print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Std CV Accuracy: {np.std(fold_accuracies):.4f}")

if __name__ == "__main__":
    main()
