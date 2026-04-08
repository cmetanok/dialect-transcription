# audio_processor.py
import soundfile as sf
import librosa
import numpy as np
import os


class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000

    def convert_to_wav(self, input_path: str, output_path: str = None) -> str:
        """Конвертация аудио в WAV формат"""
        try:
            if output_path is None:
                output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'

            # Загрузка аудио с помощью librosa
            audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)

            # Сохранение в WAV
            sf.write(output_path, audio, self.sample_rate)

            return output_path

        except Exception as e:
            print(f"Ошибка конвертации аудио: {e}")
            return input_path

    def get_audio_info(self, audio_path: str) -> dict:
        """Получение информации об аудиофайле"""
        try:
            if not os.path.exists(audio_path):
                return {"error": f"Файл не найден: {audio_path}"}

            audio, sr = librosa.load(audio_path, sr=None)

            return {
                "duration": len(audio) / sr,
                "sample_rate": sr,
                "channels": 1 if len(audio.shape) == 1 else audio.shape[1],
                "samples": len(audio),
                "max_amplitude": float(np.max(np.abs(audio))),
                "rms": float(np.sqrt(np.mean(audio ** 2)))
            }
        except Exception as e:
            return {"error": str(e)}