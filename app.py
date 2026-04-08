# app.py
import streamlit as st
from transcriber import DialectTranscriber
import os
import tempfile

# --- Настройка страницы ---
st.set_page_config(page_title="Диалектная транскрипция", layout="wide")
st.title(" Автоматическая транскрипция диалектной речи")
st.markdown("Загрузите аудиофайл, и программа создаст **орфографическую** и **фонетическую** запись.")

# --- Боковая панель с настройками ---
with st.sidebar:
    st.header("⚙️ Настройки")
    model_choice = st.selectbox(
        "Модель Whisper",
        ("tiny", "base", "small", "medium", "large"),
        index=1,  # "base" по умолчанию
        help="Чем больше модель, тем точнее, но медленнее. Для диалектов лучше 'small' или 'medium'."
    )

    st.header("🎨 Формат транскрипции")
    transcription_type = st.radio(
        "Выберите алфавит для фонетической записи:",
        ("Кириллица (русские буквы)", "Латиница (IPA)")
    )

    st.divider()
    st.caption("Проект для защиты. Используются модели: Whisper (распознавание) и собственный фонемайзер.")

# --- Основная область приложения ---
uploaded_file = st.file_uploader("📁 Выберите аудиофайл", type=["wav", "mp3", "flac", "m4a"])

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
        st.metric("Формат", uploaded_file.type)

    # Кнопка для запуска
    if st.button("🚀 Выполнить транскрипцию", type="primary"):
        with st.spinner("Идет распознавание речи. Это может занять некоторое время..."):
            # Инициализируем и запускаем транскриптор
            transcriber = DialectTranscriber(model_name=model_choice)
            result = transcriber.transcribe_speech(audio_path)

            if "error" not in result:
                st.success("✅ Транскрипция успешно завершена!")

                # Отображаем результаты
                tab1, tab2 = st.tabs(["📝 Результат", "ℹ️ О транскрипции"])
                with tab1:
                    st.subheader("Орфографическая запись")
                    st.text_area("Распознанный текст:", result['orthographic'], height=150)

                    st.subheader("Фонетическая транскрипция")
                    if transcription_type == "Кириллица (русские буквы)":
                        # TODO: Нужно добавить вызов вашего нового метода to_cyrillic
                        phonetic_result = result['phonetic']  # Заглушка, пока только IPA
                    else:
                        phonetic_result = result['phonetic']
                    st.text_area("Транскрипция:", phonetic_result, height=200)

                    # Кнопка для скачивания
                    st.download_button(
                        label="💾 Скачать результат (TXT)",
                        data=f"Орфография:\n{result['orthographic']}\n\nФонетика:\n{phonetic_result}",
                        file_name="transcription_result.txt",
                        mime="text/plain"
                    )
            else:
                st.error(f"❌ Ошибка при транскрипции: {result['error']}")

    # Удаляем временный файл
    os.unlink(audio_path)