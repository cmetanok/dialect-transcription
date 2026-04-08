# main.py
import os
import sys
import argparse
from transcriber import DialectTranscriber
from audio_processor import AudioProcessor
from config import Config
import json


def main():
    parser = argparse.ArgumentParser(description="Автоматическая транскрипция диалектной речи")
    parser.add_argument("audio_path", help="Путь к аудиофайлу")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Модель Whisper")
    parser.add_argument("--language", default="ru", help="Язык")
    parser.add_argument("--output", help="Путь для сохранения результата")
    parser.add_argument("--convert", action="store_true", help="Конвертировать аудио в WAV")

    args = parser.parse_args()

    # Создание необходимых директорий
    Config.create_dirs()

    # Инициализация процессора аудио
    audio_processor = AudioProcessor()

    # Конвертация аудио при необходимости
    audio_path = args.audio_path
    if args.convert:
        print("Конвертация аудиофайла...")
        audio_path = audio_processor.convert_to_wav(args.audio_path)
        print(f"Аудио сконвертировано: {audio_path}")

    # Получение информации об аудио
    audio_info = audio_processor.get_audio_info(audio_path)
    print("Информация об аудио:")
    for key, value in audio_info.items():
        print(f"  {key}: {value}")

    # Инициализация транскриптора
    transcriber = DialectTranscriber(model_name=args.model, language=args.language)

    # Выполнение транскрипции
    print("\nВыполнение транскрипции...")
    results = transcriber.transcribe_speech(audio_path)

    if "error" not in results:
        # Сохранение результатов
        output_path = args.output or os.path.join(Config.OUTPUT_DIR, f"transcript_{os.path.basename(audio_path)}.json")
        transcriber.save_results(results, output_path)

        # Вывод результатов
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ТРАНСКРИПЦИИ")
        print("=" * 60)
        print(f"\nОрфографическая запись:")
        print(f"{results['orthographic']}")
        print(f"\nФонетическая транскрипция:")
        print(f"{results['phonetic']}")

        if results['dialect_features']:
            print(f"\nОбнаруженные диалектные особенности:")
            for feature, info in results['dialect_features'].items():
                print(f"  - {feature}: {info['count']} вхождений")
    else:
        print(f"Ошибка: {results['error']}")


if __name__ == "__main__":
    main()