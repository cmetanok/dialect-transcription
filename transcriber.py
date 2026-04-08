# transcriber.py
import whisper
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
import re
import os
import warnings

warnings.filterwarnings("ignore")

# Импортируем дополнительные библиотеки для работы с аудио
try:
    import soundfile as sf
    import librosa

    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("Предупреждение: soundfile или librosa не установлены. Установите: pip install soundfile librosa")


class RussianPhonemizer:
    """Русский фонемайзер для детальной фонетической транскрипции"""

    def __init__(self):
        # Таблица соответствия для ЛАТИНИЦЫ (IPA - International Phonetic Alphabet)
        self.ipa_mapping = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
            'е': 'je', 'ё': 'jo', 'ж': 'ʐ', 'з': 'z', 'и': 'i',
            'й': 'j', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
            'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
            'у': 'u', 'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'tɕ',
            'ш': 'ʂ', 'щ': 'ɕː', 'ъ': '', 'ы': 'ɨ', 'ь': 'ʲ',
            'э': 'ɛ', 'ю': 'ju', 'я': 'ja'
        }

        # Таблица соответствия для КИРИЛЛИЦЫ (русские буквы)
        self.cyrillic_mapping = {
            'а': 'а', 'б': 'б', 'в': 'в', 'г': 'г', 'д': 'д',
            'е': 'йе', 'ё': 'йо', 'ж': 'ж', 'з': 'з', 'и': 'и',
            'й': 'й', 'к': 'к', 'л': 'л', 'м': 'м', 'н': 'н',
            'о': 'о', 'п': 'п', 'р': 'р', 'с': 'с', 'т': 'т',
            'у': 'у', 'ф': 'ф', 'х': 'х', 'ц': 'ц', 'ч': 'ч',
            'ш': 'ш', 'щ': 'ш\'', 'ъ': '', 'ы': 'ы', 'ь': '\'',
            'э': 'э', 'ю': 'йу', 'я': 'йа'
        }

        # Правила ассимиляции и редукции (упрощенные)
        self.assimilation_rules = [
            # Оглушение звонких на конце слова
            (r'([бвгдзж])$', r'\1̥'),
            # Смягчение согласных перед мягкими гласными
            (r'([дт])([иеёюя])', r'\1ʲ\2'),
            (r'([зс])([иеёюя])', r'\1ʲ\2'),
            (r'([лн])([иеёюя])', r'\1ʲ\2'),
        ]

    def phonemize_ipa(self, text: str) -> str:
        """
        Преобразование текста в фонетическую транскрипцию (латиница IPA)
        Пример: 'строили' -> 'stroili'
        """
        text = text.lower()
        result = []

        i = 0
        while i < len(text):
            char = text[i]

            if char in self.ipa_mapping:
                # Обработка мягкого знака (ь)
                if i + 1 < len(text) and text[i + 1] == 'ь':
                    phoneme = self.ipa_mapping[char] + 'ʲ'
                    i += 1
                else:
                    phoneme = self.ipa_mapping[char]

                # Упрощенная редукция гласных в безударной позиции
                # (для демонстрации, в полной версии нужен анализатор ударений)
                if char in 'ао' and self._is_unstressed(text, i):
                    phoneme = 'ə'
                elif char in 'еи' and self._is_unstressed(text, i):
                    phoneme = 'ɪ'

                result.append(phoneme)
            elif char == ' ':
                result.append(' ')
            elif char in '.,!?;:':
                result.append(' ')
            else:
                result.append(char)

            i += 1

        # Объединяем результат
        transcribed = ''.join(result)

        # Постобработка для формата как в примере
        transcribed = self._postprocess_ipa(transcribed, text)

        return transcribed

    def phonemize_cyrillic(self, text: str) -> str:
        """
        Преобразование текста в фонетическую транскрипцию (кириллица)
        Пример: 'строили' -> 'строил'и'
        """
        text = text.lower()
        result = []

        i = 0
        while i < len(text):
            char = text[i]

            if char in self.cyrillic_mapping:
                # Обработка мягкого знака (ь)
                if i + 1 < len(text) and text[i + 1] == 'ь':
                    phoneme = self.cyrillic_mapping[char] + '\''
                    i += 1
                else:
                    phoneme = self.cyrillic_mapping[char]

                # Упрощенная редукция гласных (для кириллицы)
                if char in 'ао' and self._is_unstressed(text, i):
                    phoneme = 'ъ'

                result.append(phoneme)
            elif char == ' ':
                result.append(' ')
            elif char in '.,!?;:':
                result.append(' ')
            else:
                result.append(char)

            i += 1

        # Объединяем результат
        transcribed = ''.join(result)

        # Постобработка для формата как в примере
        transcribed = self._postprocess_cyrillic(transcribed, text)

        return transcribed

    def _is_unstressed(self, text: str, position: int) -> bool:
        """
        Упрощенная проверка на безударность.
        В реальном приложении нужен морфологический анализатор.
        """
        # Простая эвристика: считаем, что в коротких словах (до 3 букв)
        # гласная может быть ударной, в длинных - скорее безударная
        word_start = position
        while word_start > 0 and text[word_start - 1] != ' ':
            word_start -= 1
        word_end = position
        while word_end < len(text) and text[word_end] != ' ':
            word_end += 1
        word_length = word_end - word_start

        # Если слово длинное (>4 букв), скорее всего гласная безударная
        # Это очень упрощенное правило для демонстрации
        return word_length > 4

    def _postprocess_ipa(self, phonemes: str, original_text: str) -> str:
        """
        Постобработка IPA транскрипции для форматирования
        Добавляет разделители // между предложениями и w- в начале фраз
        """
        # Разделяем на предложения по знакам препинания
        sentences = re.split(r'[.!?]+', original_text)
        phoneme_sentences = []

        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Получаем фонемы для предложения
                sentence_phonemes = self._get_sentence_phonemes_ipa(sentence)
                if sentence_phonemes:
                    # Добавляем w- в начале первого предложения
                    if i == 0 and len(sentence_phonemes) > 0:
                        sentence_phonemes = 'w-' + sentence_phonemes
                    phoneme_sentences.append(sentence_phonemes)

        # Объединяем предложения с разделителем //
        result = ' // '.join(phoneme_sentences)

        # Добавляем финальный разделитель если есть результат
        if result:
            result = result + ' //'

        return result

    def _postprocess_cyrillic(self, phonemes: str, original_text: str) -> str:
        """
        Постобработка кириллической транскрипции для форматирования
        Добавляет разделители // между предложениями и в- в начале фраз
        """
        # Разделяем на предложения по знакам препинания
        sentences = re.split(r'[.!?]+', original_text)
        phoneme_sentences = []

        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Получаем фонемы для предложения
                sentence_phonemes = self._get_sentence_phonemes_cyrillic(sentence)
                if sentence_phonemes:
                    # Добавляем в- в начале первого предложения
                    if i == 0 and len(sentence_phonemes) > 0:
                        sentence_phonemes = 'в-' + sentence_phonemes
                    phoneme_sentences.append(sentence_phonemes)

        # Объединяем предложения с разделителем //
        result = ' // '.join(phoneme_sentences)

        # Добавляем финальный разделитель если есть результат
        if result:
            result = result + ' //'

        return result

    def _get_sentence_phonemes_ipa(self, sentence: str) -> str:
        """Получение IPA фонем для отдельного предложения"""
        words = sentence.split()
        phoneme_words = []

        for word in words:
            if word:
                word_phonemes = self._word_to_phonemes_ipa(word)
                phoneme_words.append(word_phonemes)

        return ' '.join(phoneme_words)

    def _get_sentence_phonemes_cyrillic(self, sentence: str) -> str:
        """Получение кириллических фонем для отдельного предложения"""
        words = sentence.split()
        phoneme_words = []

        for word in words:
            if word:
                word_phonemes = self._word_to_phonemes_cyrillic(word)
                phoneme_words.append(word_phonemes)

        return ' '.join(phoneme_words)

    def _word_to_phonemes_ipa(self, word: str) -> str:
        """Преобразование одного слова в IPA фонемы"""
        result = []
        i = 0
        while i < len(word):
            char = word[i]
            if char in self.ipa_mapping:
                if i + 1 < len(word) and word[i + 1] == 'ь':
                    phoneme = self.ipa_mapping[char] + 'ʲ'
                    i += 1
                else:
                    phoneme = self.ipa_mapping[char]
                result.append(phoneme)
            else:
                result.append(char)
            i += 1
        return ''.join(result)

    def _word_to_phonemes_cyrillic(self, word: str) -> str:
        """Преобразование одного слова в кириллические фонемы"""
        result = []
        i = 0
        while i < len(word):
            char = word[i]
            if char in self.cyrillic_mapping:
                if i + 1 < len(word) and word[i + 1] == 'ь':
                    phoneme = self.cyrillic_mapping[char] + '\''
                    i += 1
                else:
                    phoneme = self.cyrillic_mapping[char]
                result.append(phoneme)
            else:
                result.append(char)
            i += 1
        return ''.join(result)


class DialectTranscriber:
    def __init__(self, model_name="base", language="ru"):
        """
        Инициализация транскриптора

        Args:
            model_name: название модели Whisper (tiny, base, small, medium, large)
            language: язык (по умолчанию ru - русский)
        """
        print(f"Загрузка модели {model_name}...")
        try:
            self.model = whisper.load_model(model_name)
            print(f"Модель {model_name} успешно загружена")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            print("Попытка загрузить модель 'base'...")
            self.model = whisper.load_model("base")

        self.language = language

        # Инициализация русского фонемайзера
        self.phonemizer = RussianPhonemizer()
        print("Используется встроенный русский фонемайзер")

    def transcribe_speech(self, audio_path: str) -> Dict:
        """
        Основная функция транскрипции

        Args:
            audio_path: путь к аудиофайлу

        Returns:
            словарь с результатами транскрипции
        """
        try:
            # Проверка существования файла
            if not os.path.exists(audio_path):
                raise Exception(f"Файл не найден: {audio_path}")

            print(f"Обработка файла: {audio_path}")

            # 1. Загрузка и обработка аудио
            audio = self._load_audio(audio_path)
            print("Аудио успешно загружено")

            # 2. Распознавание речи (орфографическая запись)
            orthographic = self._speech_to_text(audio)
            print(f"Распознанный текст: {orthographic}")

            # 3. Фонетическая транскрипция в двух вариантах
            phonetic_ipa = self.phonemizer.phonemize_ipa(orthographic)
            phonetic_cyrillic = self.phonemizer.phonemize_cyrillic(orthographic)

            print(f"IPA транскрипция: {phonetic_ipa[:100]}..." if len(phonetic_ipa) > 100 else f"IPA: {phonetic_ipa}")
            print(f"Кириллическая транскрипция: {phonetic_cyrillic[:100]}..." if len(
                phonetic_cyrillic) > 100 else f"Кириллица: {phonetic_cyrillic}")

            result = {
                "timestamp": datetime.now().isoformat(),
                "audio_file": audio_path,
                "orthographic": orthographic,
                "phonetic_ipa": phonetic_ipa,
                "phonetic_cyrillic": phonetic_cyrillic
            }

            # Сохраняем результаты для последующего использования
            self.last_results = result

            return result

        except Exception as e:
            print(f"Ошибка при транскрипции: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Загрузка аудиофайла с использованием разных методов"""
        try:
            # Метод 1: Загрузка через librosa (рекомендуется)
            if AUDIO_LIBS_AVAILABLE:
                try:
                    print("Загрузка через librosa...")
                    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                    # Нормализация аудио
                    if np.max(np.abs(audio)) > 0:
                        audio = audio / np.max(np.abs(audio))
                    print(f"Аудио загружено через librosa, длина: {len(audio)} samples")
                    return audio
                except Exception as e:
                    print(f"Ошибка загрузки через librosa: {e}")

            # Метод 2: Загрузка через soundfile (только для wav)
            if audio_path.lower().endswith('.wav'):
                try:
                    print("Загрузка через soundfile...")
                    audio, sr = sf.read(audio_path)
                    if sr != 16000:
                        import scipy.signal
                        audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    # Нормализация
                    if np.max(np.abs(audio)) > 0:
                        audio = audio / np.max(np.abs(audio))
                    print(f"Аудио загружено через soundfile, длина: {len(audio)} samples")
                    return audio
                except Exception as e:
                    print(f"Ошибка загрузки через soundfile: {e}")

            # Метод 3: Загрузка через whisper (требует ffmpeg)
            try:
                print("Загрузка через whisper...")
                audio = whisper.load_audio(audio_path)
                audio = whisper.pad_or_trim(audio)
                print(f"Аудио загружено через whisper")
                return audio
            except Exception as e:
                print(f"Ошибка загрузки через whisper: {e}")

            raise Exception("Не удалось загрузить аудио ни одним из методов")

        except Exception as e:
            raise Exception(f"Ошибка загрузки аудио: {e}")

    def _speech_to_text(self, audio: np.ndarray) -> str:
        """Распознавание речи в текст"""
        try:
            # Транскрипция через Whisper
            result = self.model.transcribe(
                audio,
                language=self.language,
                verbose=False,
                task="transcribe"
            )
            text = result["text"].strip()

            if not text:
                print("Предупреждение: текст не распознан")
                return ""

            # Нормализация текста
            text = self._normalize_text(text)

            return text
        except Exception as e:
            raise Exception(f"Ошибка распознавания речи: {e}")

    def _normalize_text(self, text: str) -> str:
        """Нормализация текста"""
        # Приведение к нижнему регистру
        text = text.lower()

        # Удаление лишних пробелов
        text = ' '.join(text.split())

        return text

    def save_results(self, results: Dict, output_path: str = None):
        """Сохранение результатов в файл"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"transcription_{timestamp}.json"

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Результаты сохранены в {output_path}")

            # Создание текстового отчета
            txt_path = output_path.replace('.json', '.txt')
            self._save_text_report(results, txt_path)
        except Exception as e:
            print(f"Ошибка при сохранении результатов: {e}")

    def _save_text_report(self, results: Dict, output_path: str):
        """Сохранение текстового отчета"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ОТЧЕТ О ТРАНСКРИПЦИИ ДИАЛЕКТНОЙ РЕЧИ\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Аудиофайл: {results.get('audio_file', 'N/A')}\n")
                f.write(f"Время обработки: {results.get('timestamp', 'N/A')}\n\n")

                f.write("ОРФОГРАФИЧЕСКАЯ ЗАПИСЬ:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{results['orthographic']}\n\n")

                f.write("ФОНЕТИЧЕСКАЯ ТРАНСКРИПЦИЯ (IPA - латиница):\n")
                f.write("-" * 80 + "\n")
                f.write(f"{results.get('phonetic_ipa', 'Нет данных')}\n\n")

                f.write("ФОНЕТИЧЕСКАЯ ТРАНСКРИПЦИЯ (КИРИЛЛИЦА):\n")
                f.write("-" * 80 + "\n")
                f.write(f"{results.get('phonetic_cyrillic', 'Нет данных')}\n")

        except Exception as e:
            print(f"Ошибка при сохранении текстового отчета: {e}")