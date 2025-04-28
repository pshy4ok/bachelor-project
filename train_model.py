import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers
from keras.api.utils import to_categorical
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Налаштування параметрів
SR = 22050
SEGMENT_DURATION = 5
MFCC_NUM = 64
CHROMA_NUM = 12
MAX_PAD_LEN = 862
GENRE_DIR = "data"

GENRE_LIST = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
genre_dict = {i: genre for i, genre in enumerate(GENRE_LIST)}


def augment_audio(audio, sr):
    try:
        noise = np.random.randn(len(audio))
        audio_noise = audio + 0.005 * noise
        audio_pitch = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=2)
        return [audio, audio_noise, audio_pitch]
    except Exception as e:
        logger.error(f"Error in audio augmentation: {str(e)}")
        return [audio]


def extract_features(audio, sr=SR, segment_duration=SEGMENT_DURATION):
    try:
        logger.info("Extracting features from audio")
        segment_samples = segment_duration * sr
        num_segments = int(len(audio) // segment_samples)
        if num_segments == 0:
            logger.warning("Audio too short for segmentation")
            return None

        mfcc_features = []
        chroma_features = []
        spectral_centroids = []

        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = audio[start:end]

            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=MFCC_NUM)
            mfccs = librosa.util.normalize(mfccs)
            if mfccs.shape[1] < MAX_PAD_LEN:
                pad_width = MAX_PAD_LEN - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :MAX_PAD_LEN]
            mfcc_features.append(mfccs)

            chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=CHROMA_NUM)
            chroma = librosa.util.normalize(chroma)
            if chroma.shape[1] < MAX_PAD_LEN:
                pad_width = MAX_PAD_LEN - chroma.shape[1]
                chroma = np.pad(chroma, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                chroma = chroma[:, :MAX_PAD_LEN]
            chroma_features.append(chroma)

            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
            centroid = librosa.util.normalize(centroid)
            if centroid.shape[1] < MAX_PAD_LEN:
                pad_width = MAX_PAD_LEN - centroid.shape[1]
                centroid = np.pad(centroid, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                centroid = centroid[:, :MAX_PAD_LEN]
            spectral_centroids.append(centroid)

        mfcc_agg = np.stack([np.mean(mfcc_features, axis=0),
                             np.max(mfcc_features, axis=0),
                             np.std(mfcc_features, axis=0)], axis=-1)
        chroma_mean = np.mean(chroma_features, axis=0)
        chroma_padded = np.pad(chroma_mean, ((0, 64 - CHROMA_NUM), (0, 0)), mode='constant')
        centroid_mean = np.mean(spectral_centroids, axis=0)
        centroid_padded = np.pad(centroid_mean, ((0, 64 - 1), (0, 0)), mode='constant')

        combined_features = np.stack([mfcc_agg[:, :, 0], mfcc_agg[:, :, 1], mfcc_agg[:, :, 2],
                                      chroma_padded, centroid_padded], axis=-1)
        logger.info("Features extracted successfully")
        return combined_features
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None


def load_data():
    features = []
    labels = []
    logger.info(f"Loading data from {GENRE_DIR}")
    if not os.path.exists(GENRE_DIR):
        logger.error(f"Directory {GENRE_DIR} does not exist")
        return np.array([]), np.array([])

    for genre_label, genre in genre_dict.items():
        genre_folder = os.path.join(GENRE_DIR, genre)
        if not os.path.exists(genre_folder):
            logger.warning(f"Genre folder {genre_folder} does not exist")
            continue

        logger.info(f"Processing genre: {genre}")
        files = os.listdir(genre_folder)
        for file in files:
            file_path = os.path.join(genre_folder, file)
            logger.info(f"Processing file: {file_path}")
            try:
                audio, sr = librosa.load(file_path, sr=SR, mono=True)
                augmented_audios = augment_audio(audio, sr)
                for idx, aug_audio in enumerate(augmented_audios):
                    logger.info(f"Processing augmented audio {idx + 1}/{len(augmented_audios)}")
                    feature = extract_features(aug_audio)
                    if feature is not None:
                        features.append(feature)
                        labels.append(genre_label)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                continue

    features = np.array(features)
    labels = np.array(labels)
    logger.info(f"Loaded {len(features)} samples with {len(labels)} labels")
    return features, labels


def create_model(input_shape, num_genres):
    inputs = layers.Input(shape=input_shape)

    # Residual Block 1
    x = layers.Conv2D(32, (5, 5), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    shortcut = x
    x = layers.Conv2D(32, (5, 5), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])

    # Residual Block 2
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    shortcut = layers.Conv2D(64, (1, 1), padding='same')(shortcut)
    shortcut = layers.MaxPooling2D((2, 2))(shortcut)
    x = layers.Add()([shortcut, x])

    # Residual Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    shortcut = layers.Conv2D(128, (1, 1), padding='same')(shortcut)
    shortcut = layers.MaxPooling2D((2, 2))(shortcut)
    x = layers.Add()([shortcut, x])

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_genres, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    logger.info("Starting training process")
    features, labels = load_data()
    if len(features) == 0:
        logger.error("No valid audio files processed. Check data directory and file formats.")
    else:
        logger.info("Splitting data into train and validation sets")
        X_train, X_val, y_train, y_val = train_test_split(features, to_categorical(labels), test_size=0.2,
                                                          random_state=42)
        input_shape = (64, 862, 5)
        num_genres = len(genre_dict)

        logger.info("Creating model")
        model = create_model(input_shape, num_genres)

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
        ]

        logger.info("Starting model training")
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=64, callbacks=callbacks)

        logger.info("Evaluating model")
        y_pred_proba = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
        logger.info(f"AUC-ROC: {auc:.4f}")

        logger.info("Saving model")
        model.save('model.keras')