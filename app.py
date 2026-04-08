# app.py
import streamlit as st
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

# Импорт вашего модуля
from transcriber import DialectTranscriber

# --- Настройка страницы ---
st.set_page_config(
    page_title="Диалектная транскрипция",
    page_icon="🗣️",
    layout="wide"
)

st.title("🗣️ Автоматическая транскрипция диалектной речи")
st.markdown("Загрузите аудиофайл, и программа создаст **орфографическую** и **фонетическую** запись.")

# --- Боковая панель с настройками ---
with st.sidebar:
    st.header("⚙️ Настройки")

    model_choice = st.selectbox(
        "Модель Whisper",
        ("base", "tiny", "small", "medium"),
        index=0,
        help="Чем больше модель, тем точнее, но медленнее. 'base' - оптимальный выбор."
    )

    st.header("🎨 Формат транскрипции")
    transcription_type = st.radio(
        "Выберите алфавит для фонетической записи:",
        ("Кириллица (русские буквы)", "Латиница (IPA)"),
        index=0
    )

    st.divider()
    st.caption("📌 **Используемые технологии:**")
    st.caption("- Whisper (распознавание речи)")
    st.caption("- Собственный русский фонемайзер")
    st.caption("- Streamlit (веб-интерфейс)")

# --- Основная область приложения ---
uploaded_file = st.file_uploader(
    "📁 Выберите аудиофайл",
    type=["wav", "mp3", "flac", "m4a"],
    help="Поддерживаются форматы: WAV, MP3, FLAC, M4A"
)

if uploaded_file is not None:
    # Сохраняем загруженный файл во временную директорию
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name

    # Показываем информацию о файле
    col1, col2 = st.columns(2)
    with col1:
        st.audio(uploaded_file, format='audio/wav')
    with col2:
        st.metric("Размер файла", f"{uploaded_file.size / (1024 * 1024):.2f} MB")
        st.metric("Формат", uploaded_file.type.split('/')[-1].upper())

    # Кнопка для запуска
    if st.button("🚀 Выполнить транскрипцию", type="primary"):
        with st.spinner("🔄 Идет распознавание речи. Это может занять 10-30 секунд..."):
            try:
                # Инициализируем и запускаем транскриптор
                transcriber = DialectTranscriber(model_name=model_choice)
                result = transcriber.transcribe_speech(audio_path)

                if "error" not in result:
                    st.success("✅ Транскрипция успешно завершена!")

                    # Отображаем результаты
                    tab1, tab2 = st.tabs(["📝 Результат транскрипции", "ℹ️ О транскрипции"])

                    with tab1:
                        st.subheader("Орфографическая запись")
                        st.text_area(
                            "Распознанный текст:",
                            result['orthographic'],
                            height=150,
                            key="ortho"
                        )

                        st.subheader("Фонетическая транскрипция")
                        if transcription_type == "Кириллица (русские буквы)":
                            phonetic_result = result.get('phonetic_cyrillic', 'Нет данных')
                            st.text_area(
                                "Транскрипция (кириллица):",
                                phonetic_result,
                                height=200,
                                key="cyr"
                            )
                        else:
                            phonetic_result = result.get('phonetic_ipa', 'Нет данных')
                            st.text_area(
                                "Транскрипция (IPA):",
                                phonetic_result,
                                height=200,
                                key="ipa"
                            )

                        # Кнопка для скачивания
                        download_data = f"""Орфографическая запись:
{result['orthographic']}

Фонетическая транскрипция ({transcription_type}):
{phonetic_result}

--- 
Создано программой диалектной транскрипции
Время: {result['timestamp']}
"""
                        st.download_button(
                            label="💾 Скачать результат (TXT)",
                            data=download_data,
                            file_name="transcription_result.txt",
                            mime="text/plain"
                        )

                    with tab2:
                        st.markdown("""
                        ### Как работает транскрипция?

                        1. **Распознавание речи** выполняется нейросетью **Whisper** от OpenAI
                        2. **Фонетическая транскрипция** создается собственным алгоритмом

                        ### Обозначения в транскрипции:

                        | Символ | Значение |
                        |--------|----------|
                        | `//` | Длинная пауза (граница предложения) |
                        | `/` | Короткая пауза |
                        | `w-` / `в-` | Губно-губной звук в начале фразы |
                        | `'` / `ʲ` | Мягкость (палатализация) согласного |

                        ### Советы:
                        - Используйте аудио с четкой речью
                        - Для длинных записей (более 1 минуты) выбирайте модель "tiny"
                        - Для лучшего качества используйте "small" или "medium"
                        """)
                else:
                    st.error(f"❌ Ошибка при транскрипции: {result['error']}")

            except Exception as e:
                st.error(f"❌ Критическая ошибка: {str(e)}")
                st.info("Попробуйте выбрать другую модель или загрузить файл меньшего размера")

    # Удаляем временный файл
    try:
        os.unlink(audio_path)
    except:
        pass

# Добавляем информацию в футер
st.divider()
st.caption("© 2024 - Диалектная транскрипция | Для защиты курсовой работы")