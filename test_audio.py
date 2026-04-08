# create_test_audio.py
import numpy as np
import soundfile as sf
import os

# Создаем тестовое аудио с речью (простой сигнал)
sample_rate = 16000
duration = 3  # секунды
t = np.linspace(0, duration, int(sample_rate * duration))

# Создаем простой тон
frequency = 440  # нота Ля
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

# Добавляем шум для реалистичности
noise = np.random.normal(0, 0.05, len(audio))
audio = audio + noise

# Сохраняем
output_path = "test_audio.wav"
sf.write(output_path, audio, sample_rate)
print(f"Тестовый аудиофайл создан: {output_path}")