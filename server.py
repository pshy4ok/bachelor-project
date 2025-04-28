from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
from fastapi.middleware.cors import CORSMiddleware
import io
from pydub import AudioSegment
import tempfile
import os

app = FastAPI()

# Налаштування CORS для Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Налаштування параметрів
SR = 22050
SEGMENT_DURATION = 5
MFCC_NUM = 64
CHROMA_NUM = 12
MAX_PAD_LEN = 862

# Завантаження моделі та жанрів
MODEL_PATH = "best_model.keras"
GENRE_LIST = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
genre_dict = {i: genre for i, genre in enumerate(GENRE_LIST)}
model = tf.keras.models.load_model(MODEL_PATH)

# Функція витягнення ознак
def extract_features(audio, sr=SR, segment_duration=SEGMENT_DURATION):
    try:
        audio = librosa.util.normalize(audio.flatten())
        segment_samples = segment_duration * sr
        num_segments = int(len(audio) // segment_samples)
        if num_segments == 0:
            raise ValueError("Аудіо занадто коротке для обробки")

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
        return combined_features
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Помилка обробки аудіо: {str(e)}")

# Ендпоінт для передбачення з файлу
@app.post("/predict/")
async def predict_genre(file: UploadFile = File(...)):
    print("Receiving file:", file.filename)
    try:
        # Зчитуємо вміст файлу
        audio_bytes = await file.read()
        print(f"File size: {len(audio_bytes)} bytes")

        # Спробуємо завантажити через pydub і конвертувати в WAV
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_segment = audio_segment.set_frame_rate(SR).set_channels(1)  # Конвертуємо до моно і SR=22050
            audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0  # Нормалізація до [-1, 1]
            print("File converted successfully with pydub")
        except Exception as e:
            print(f"Pydub failed: {str(e)}. Falling back to librosa.")
            # Якщо pydub не спрацював, пробуємо librosa
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)

        print("File loaded successfully")
        if len(audio) == 0:
            raise ValueError("Аудіофайл порожній")

        features = extract_features(audio)
        features = features[np.newaxis, ...]
        predictions = model.predict(features)
        predicted_idx = np.argmax(predictions)
        genre = genre_dict[predicted_idx]
        confidence = float(predictions[0][predicted_idx]) * 100
        print("Prediction made:", genre, confidence)
        return {"genre": genre, "confidence": confidence}
    except Exception as e:
        print("Error processing file:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

# Ендпоінт для передбачення з мікрофона
@app.post("/predict/microphone/")
async def predict_from_microphone(duration: int = 16):
    try:
        print("Запис аудіо з мікрофона...")
        audio = sd.rec(int(duration * SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        features = extract_features(audio)
        features = features[np.newaxis, ...]
        predictions = model.predict(features)
        predicted_idx = np.argmax(predictions)
        genre = genre_dict[predicted_idx]
        confidence = float(predictions[0][predicted_idx]) * 100
        print("Prediction made:", genre, confidence)
        return {"genre": genre, "confidence": confidence}
    except Exception as e:
        print("Error processing microphone input:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)