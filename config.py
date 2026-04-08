# config.py
import os


class Config:
    # Модели
    WHISPER_MODEL = "base"

    # Пути
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR = os.path.join(BASE_DIR, "audio")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    # Настройки аудио
    SAMPLE_RATE = 16000
    CHANNELS = 1

    @staticmethod
    def create_dirs():
        os.makedirs(Config.AUDIO_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)