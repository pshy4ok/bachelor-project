import time
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread


# Завантаження жанрів
def load_genre_dict(file_path):
    with open(file_path, 'r') as f:
        genres = f.read().splitlines()
    return {i: genre for i, genre in enumerate(genres)}


# Параметри аудіо
SR = 22050
DURATION = 5
FRAME_LENGTH = SR * DURATION
MFCC_NUM = 64
MAX_PAD_LEN = 862

# Черга для аудіо
audio_queue = queue.Queue()

# Завантаження жанрів і моделі
genre_dict = load_genre_dict('genres/genres.txt')
model = tf.keras.models.load_model('model.keras')


def extract_features(audio, sr=SR):
    try:
        # Переконуємося, що аудіо одновимірний
        if len(audio.shape) > 1:
            audio = audio.flatten()

        # Нормалізація аудіо
        audio = librosa.util.normalize(audio)

        # Оптимізовані параметри для MFCC
        n_fft = 2048  # Збільшено розмір вікна
        hop_length = 512  # Збільшено hop_length
        n_mels = 128  # Збільшено кількість MEL-фільтрів

        # Обчислення MFCC з оптимізованими параметрами
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=MFCC_NUM,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=20,
            fmax=sr / 2.0  # Використовуємо половину частоти дискретизації як fmax
        )

        # Нормалізація MFCC
        mfccs = librosa.util.normalize(mfccs)

        # Обробка розмірності
        if mfccs.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]

        return mfccs

    except Exception as e:
        messagebox.showerror("Помилка", f"Помилка обробки аудіо: {str(e)}")
        return None


# Функція передбачення жанру
def predict_genre(audio):
    features = extract_features(audio)
    if features is None:
        return

    features = features.reshape(1, MFCC_NUM, MAX_PAD_LEN, 1)
    predictions = model.predict(features)
    predicted_genre_idx = np.argmax(predictions)
    predicted_genre = genre_dict[predicted_genre_idx]

    # Отримуємо відсоток впевненості
    confidence = float(predictions[0][predicted_genre_idx]) * 100

    genre_label.config(text=f"🎶 Жанр: {predicted_genre}\nВпевненість: {confidence:.1f}%")


# Вибір файлу та передбачення жанру
def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])
    if file_path:
        try:
            genre_label.config(text="⏳ Обробка аудіо...")
            root.update()
            audio, sr = librosa.load(file_path, sr=SR, mono=True)
            predict_genre(audio)
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося завантажити файл: {str(e)}")
            genre_label.config(text="❌ Помилка обробки файлу")


def callback(indata, frames, time, status):
    if status:
        print(status)
    # Конвертуємо в float32 та нормалізуємо
    audio_data = indata.astype(np.float32)
    audio_data = librosa.util.normalize(audio_data)
    audio_queue.put(audio_data)


def listen_and_predict():
    def run():
        try:
            genre_label.config(text="🎤 Запис...")
            audio_data = []
            start_time = time.time()

            # Оновлені параметри запису
            with sd.InputStream(samplerate=SR,
                                channels=1,
                                dtype=np.float32,
                                blocksize=2048,
                                callback=callback):
                while True:
                    try:
                        data = audio_queue.get(timeout=1.0)
                        audio_data.append(data)
                        elapsed_time = time.time() - start_time
                        if elapsed_time >= 30:  # Зменшено до 5 секунд
                            break
                    except queue.Empty:
                        continue

            if audio_data:
                genre_label.config(text="⏳ Обробка аудіо...")
                root.update()
                # Об'єднуємо та нормалізуємо
                audio_data = np.concatenate(audio_data, axis=0)
                audio_data = librosa.util.normalize(audio_data.flatten())
                predict_genre(audio_data)
            else:
                genre_label.config(text="❌ Помилка запису аудіо")

        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка запису аудіо: {str(e)}")
            genre_label.config(text="❌ Помилка запису аудіо")

    Thread(target=run, daemon=True).start()


# Інтерфейс
root = tk.Tk()
root.title("Визначення жанру аудіо")
root.geometry("415x385")

# Стилізація вікна
root.configure(bg='#f0f0f0')  # Світло-сірий фон

# Заголовок
header_label = tk.Label(
    root,
    text="🎵 Визначення жанру музики",
    font=("Arial", 16, "bold"),
    bg='#f0f0f0',
    pady=10
)
header_label.pack()

# Підказка
tk.Label(
    root,
    text="Оберіть джерело аудіо:",
    font=("Arial", 12),
    bg='#f0f0f0'
).pack(pady=5)

# Контейнер для кнопок
button_frame = tk.Frame(root, bg='#f0f0f0')
button_frame.pack(pady=10)

# Кнопки
btn_file = tk.Button(
    button_frame,
    text="📂 Завантажити аудіо",
    command=choose_file,
    font=("Arial", 11),
    bg='#4CAF50',  # Зелений
    fg='white',
    padx=20,
    pady=5
)
btn_file.pack(pady=5)

btn_mic = tk.Button(
    button_frame,
    text="🎤 Визначити з мікрофона",
    command=listen_and_predict,
    font=("Arial", 11),
    bg='#2196F3',  # Синій
    fg='white',
    padx=20,
    pady=5
)
btn_mic.pack(pady=5)

# Мітка для результату
genre_label = tk.Label(
    root,
    text="",
    font=("Arial", 14, "bold"),
    fg="#333333",
    bg='#f0f0f0'
)
genre_label.pack(pady=20)

# Інформаційна мітка
info_label = tk.Label(
    root,
    text="Запис триває 5 секунд\nПідтримуються формати: WAV, MP3, FLAC",
    font=("Arial", 10),
    fg="#666666",
    bg='#f0f0f0'
)
info_label.pack(pady=10)

root.mainloop()