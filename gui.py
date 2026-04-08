# gui.py итог
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import sys

# Добавляем путь к проекту
sys.path.append(os.path.dirname(__file__))

try:
    from transcriber import DialectTranscriber
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Пожалуйста, установите зависимости: pip install openai-whisper")
    sys.exit(1)

from audio_processor import AudioProcessor
from config import Config
import json


class TranscriptionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Диалектная транскрипция речи")
        self.root.geometry("1000x750")

        self.transcriber = None
        self.current_audio = None

        self.create_widgets()

    def create_widgets(self):
        # Создаем вкладки - используем ttk.Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Вкладка транскрипции
        self.transcription_frame = tk.Frame(self.notebook)
        self.notebook.add(self.transcription_frame, text="Транскрипция")

        # Вкладка руководства
        self.help_frame = tk.Frame(self.notebook)
        self.notebook.add(self.help_frame, text="Руководство пользователя")

        self.create_transcription_tab()
        self.create_help_tab()

    def create_transcription_tab(self):
        # Верхняя панель
        top_frame = tk.Frame(self.transcription_frame)
        top_frame.pack(pady=10)

        tk.Button(top_frame, text="Выбрать аудиофайл", command=self.load_audio,
                  bg="lightblue", width=20, height=2).pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame, text="Выполнить транскрипцию", command=self.run_transcription,
                  bg="lightgreen", width=20, height=2).pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame, text="Сохранить результат", command=self.save_results,
                  bg="lightyellow", width=20, height=2).pack(side=tk.LEFT, padx=5)

        # Настройки
        settings_frame = tk.LabelFrame(self.transcription_frame, text="Настройки", padx=5, pady=5)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(settings_frame, text="Модель Whisper:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, padx=5)
        self.model_var = tk.StringVar(value="base")
        model_menu = ttk.OptionMenu(settings_frame, self.model_var, "base", "tiny", "base", "small", "medium", "large")
        model_menu.grid(row=0, column=1, sticky=tk.W, padx=5)

        tk.Label(settings_frame, text="Язык:", font=("Arial", 10)).grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        self.lang_var = tk.StringVar(value="ru")
        tk.Entry(settings_frame, textvariable=self.lang_var, width=5).grid(row=0, column=3, sticky=tk.W)

        # Информация о файле
        self.file_info = tk.Label(self.transcription_frame, text="Файл не выбран",
                                  fg="gray", font=("Arial", 9))
        self.file_info.pack(pady=5)

        # Область вывода
        output_frame = tk.Frame(self.transcription_frame)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Орфографическая запись
        tk.Label(output_frame, text="Орфографическая запись:", font=("Arial", 12, "bold"),
                 fg="darkblue").pack(anchor=tk.W)
        self.ortho_text = scrolledtext.ScrolledText(output_frame, height=8, wrap=tk.WORD,
                                                    font=("Courier", 10))
        self.ortho_text.pack(fill=tk.X, pady=(0, 10))

        # Фонетическая транскрипция
        tk.Label(output_frame, text="Фонетическая транскрипция:", font=("Arial", 12, "bold"),
                 fg="darkgreen").pack(anchor=tk.W)
        self.phone_text = scrolledtext.ScrolledText(output_frame, height=12, wrap=tk.WORD,
                                                    font=("Courier", 10))
        self.phone_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Строка состояния
        self.status_bar = tk.Label(self.transcription_frame, text="Готов", bd=1,
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_help_tab(self):
        """Создание вкладки с руководством пользователя"""
        # Создаем текстовое поле с прокруткой
        help_text = scrolledtext.ScrolledText(self.help_frame, wrap=tk.WORD,
                                              font=("Arial", 10), padx=10, pady=10)
        help_text.pack(fill=tk.BOTH, expand=True)

        # Содержание руководства
        help_content = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    РУКОВОДСТВО ПОЛЬЗОВАТЕЛЯ                                 ║
║              Автоматическая транскрипция диалектной речи                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. О ПРОГРАММЕ
═══════════════════════════════════════════════════════════════════════════════
Программа предназначена для автоматической транскрипции диалектной звучащей 
речи. Она создает два типа записей:
• Орфографическая запись - текст в стандартном написании
• Фонетическая транскрипция - детальная запись произношения

2. УСТАНОВКА И ЗАПУСК
═══════════════════════════════════════════════════════════════════════════════
1. Установите зависимости:
   pip install openai-whisper soundfile librosa scipy

2. Запустите программу:
   python gui.py

3. ИНТЕРФЕЙС ПРОГРАММЫ
═══════════════════════════════════════════════════════════════════════════════
Главное окно содержит следующие элементы:

• Кнопка "Выбрать аудиофайл" - для загрузки аудиофайла
• Кнопка "Выполнить транскрипцию" - запуск процесса транскрипции
• Кнопка "Сохранить результат" - сохранение результатов в файл
• Выбор модели Whisper - выбор размера модели (tiny, base, small, medium, large)
• Выбор языка - указание языка распознавания (по умолчанию ru)
• Поле для орфографической записи - отображает распознанный текст
• Поле для фонетической транскрипции - отображает детальную транскрипцию

4. ПОДДЕРЖИВАЕМЫЕ ФОРМАТЫ АУДИО
═══════════════════════════════════════════════════════════════════════════════
Программа поддерживает следующие форматы:
• WAV (.wav)
• MP3 (.mp3)
• FLAC (.flac)
• M4A (.m4a)

Рекомендуется использовать файлы с частотой дискретизации 16 кГц.

5. МОДЕЛИ WHISPER
═══════════════════════════════════════════════════════════════════════════════
Доступные модели (от быстрой к точной):
• tiny     - самая быстрая, наименьшая точность
• base     - хороший баланс скорости и качества (рекомендуется)
• small    - высокая точность, средняя скорость
• medium   - очень высокая точность, медленная
• large    - максимальная точность, самая медленная

6. ФОРМАТ ФОНЕТИЧЕСКОЙ ТРАНСКРИПЦИИ
═══════════════════════════════════════════════════════════════════════════════
Программа использует систему фонетической транскрипции на основе IPA 
(International Phonetic Alphabet). Пример транскрипции:

Орфографическая запись:
у нас строились куряни в пять комнатей две комнати их и не строили

Фонетическая транскрипция:
w-у-нас строил’и кур’ан’и w-п’ат’ камнат’ей // дв’е комнат’и / их и н’и строил’и //

Обозначения:
• / - короткая пауза
• // - длинная пауза (граница предложения)
• w- - губно-губной аппроксимант в начале фразы
• ʲ - палатализация (мягкость) согласного
• ̠ - ретрофлексный согласный

7. ПОШАГОВАЯ ИНСТРУКЦИЯ
═══════════════════════════════════════════════════════════════════════════════

Шаг 1: Выбор аудиофайла
───────────────────────────────────────────────────────────────────────────────
Нажмите кнопку "Выбрать аудиофайл" и укажите путь к аудиофайлу с диалектной 
речью. Поддерживаются форматы WAV, MP3, FLAC, M4A.

Шаг 2: Настройка параметров
───────────────────────────────────────────────────────────────────────────────
Выберите модель Whisper (рекомендуется "base") и укажите язык (по умолчанию "ru").

Шаг 3: Запуск транскрипции
───────────────────────────────────────────────────────────────────────────────
Нажмите кнопку "Выполнить транскрипцию". Процесс может занять от нескольких 
секунд до нескольких минут в зависимости от длительности аудио и выбранной модели.

Шаг 4: Просмотр результатов
───────────────────────────────────────────────────────────────────────────────
После завершения транскрипции:
• В верхнем поле отобразится орфографическая запись
• В нижнем поле - фонетическая транскрипция

Шаг 5: Сохранение результатов
───────────────────────────────────────────────────────────────────────────────
Нажмите кнопку "Сохранить результат". Файл будет сохранен в формате JSON и TXT.

8. ТИПИЧНЫЕ ПРОБЛЕМЫ И ИХ РЕШЕНИЕ
═══════════════════════════════════════════════════════════════════════════════

Проблема: Не загружается аудиофайл
Решение: Убедитесь, что файл существует и имеет поддерживаемый формат.

Проблема: Низкое качество распознавания
Решение: 
• Используйте аудио с четкой речью и минимальным шумом
• Выберите более точную модель (small, medium или large)
• Убедитесь, что язык выбран правильно

Проблема: Долгая обработка
Решение: 
• Выберите модель tiny или base для ускорения
• Используйте более короткие аудиофайлы
• Убедитесь, что компьютер имеет достаточную мощность

9. ФОРМАТ ВЫХОДНЫХ ФАЙЛОВ
═══════════════════════════════════════════════════════════════════════════════

JSON файл содержит структурированные данные:
{
  "timestamp": "время обработки",
  "audio_file": "путь к аудиофайлу",
  "orthographic": "орфографическая запись",
  "phonetic": "фонетическая транскрипция"
}

TXT файл содержит читаемый отчет с обеими формами записи.

10. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ
═══════════════════════════════════════════════════════════════════════════════
• Python 3.7 или выше
• 4+ ГБ оперативной памяти (рекомендуется 8+ ГБ)
• 2+ ГБ свободного места на диске
• Для моделей medium/large рекомендуется наличие видеокарты с CUDA

11. ПОДДЕРЖКА И ОБРАТНАЯ СВЯЗЬ
═══════════════════════════════════════════════════════════════════════════════
По всем вопросам и предложениям обращайтесь к разработчику.

═══════════════════════════════════════════════════════════════════════════════
                          © 2024 Диалектная транскрипция
═══════════════════════════════════════════════════════════════════════════════
        """

        help_text.insert(1.0, help_content)
        help_text.config(state=tk.DISABLED)  # Запрещаем редактирование

    def load_audio(self):
        file_path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.m4a *.WAV *.MP3"),
                ("WAV files", "*.wav *.WAV"),
                ("MP3 files", "*.mp3 *.MP3"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_audio = file_path
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # В МБ
            self.file_info.config(text=f"Файл: {file_name} ({file_size:.2f} МБ)", fg="green")
            self.status_bar.config(text=f"Загружен: {file_name}")
            messagebox.showinfo("Успех", f"Аудиофайл загружен:\n{file_path}")

    def run_transcription(self):
        if not self.current_audio:
            messagebox.showwarning("Внимание", "Сначала выберите аудиофайл!")
            return

        # Запуск в отдельном потоке
        threading.Thread(target=self._transcribe_worker, daemon=True).start()

    def _transcribe_worker(self):
        try:
            self.status_bar.config(text="Выполняется транскрипция...")

            # Инициализация транскриптора
            self.transcriber = DialectTranscriber(
                model_name=self.model_var.get(),
                language=self.lang_var.get()
            )

            # Транскрипция
            results = self.transcriber.transcribe_speech(self.current_audio)

            if "error" not in results:
                # Обновление GUI в главном потоке
                self.root.after(0, self.update_results, results)
                self.status_bar.config(text="Транскрипция завершена")
                self.root.after(0, lambda: messagebox.showinfo("Успех", "Транскрипция успешно завершена!"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", results["error"]))
                self.status_bar.config(text="Ошибка транскрипции")

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
            self.status_bar.config(text="Ошибка транскрипции")

    def update_results(self, results):
        # Очистка полей
        self.ortho_text.delete(1.0, tk.END)
        self.phone_text.delete(1.0, tk.END)

        # Заполнение результатов
        self.ortho_text.insert(1.0, results['orthographic'])

        if results.get('phonetic'):
            self.phone_text.insert(1.0, results['phonetic'])
        else:
            self.phone_text.insert(1.0, "Фонетическая транскрипция недоступна")

    def save_results(self):
        if self.transcriber and hasattr(self.transcriber, 'last_results'):
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                self.transcriber.save_results(self.transcriber.last_results, file_path)
                messagebox.showinfo("Успех", f"Результаты сохранены в:\n{file_path}")


if __name__ == "__main__":
    # Создание необходимых директорий
    Config.create_dirs()

    root = tk.Tk()
    app = TranscriptionGUI(root)
    root.mainloop()