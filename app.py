import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="РГР: Инференс моделей ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "models"))

MODEL_FILES = {
    "Классическая модель (Poly Regression + ElasticNet)": "PolyRegression_ElasticNet_with_Optuna.pkl",
    "Ансамблевая модель (Бэггинг)": "BaggingRegressor_with_GridSearchCV.pkl",
    "Ансамблевая модель (Бустинг)": "GradientBoostingRegressor_with_Optuna.pkl",
    "Продвинутый градиентный бустинг (CatBoost)": "CatBoostRegressor_with_Optuna.pkl",
    "Ансамблевая модель (Стэкинг)": "StackingRegressor_with_Optuna.pkl",
    "Глубокая полносвязная нейронная сеть (MLP)": "MLPRegressor_with_Optuna_by_lbfgs.pkl"
}

@st.cache_resource
def load_ml_pipeline(model_display_name):
    file_name = MODEL_FILES[model_display_name]
    full_path = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    return None

st.sidebar.title("Навигация по РГР")
page = st.sidebar.radio(
    "Перейти на страницу:",
    ["Страница 1: О разработчике", 
     "Страница 2: Описание набора данных", 
     "Страница 3: Визуализация зависимостей", 
     "Страница 4: Инференс моделей ML"]
)

if page == "Страница 1: О разработчике":
    st.title("Расчетно-графическая работа")
    st.subheader("Тема: «Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных»")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        photo_path = os.path.join(BASE_DIR, "assets/developer_photo.jpg")
        if os.path.exists(photo_path):
            st.image(photo_path, caption="Фотография разработчика", width=350)
        else:
            st.info("Фото не найдено")
            
    with col2:
        st.markdown("Сведения о разработчике")
        st.write("**ФИО:** Потапов Ярослав Игоревич")
        st.write("**Учебная группа:** ФИТ-242/1")
        st.write("**Учебное заведение:** Омский государственный технический университет (ОмГТУ)")
        st.write("**Семестр:** 4 семестр, 2026 г.")
        st.write("**Дисциплина:** Машинное обучение и большие данные")



elif page == "Страница 2: Описание набора данных":
    st.title("Предметная область и описание датасета")
    st.markdown("""
    ### Датасет: Прогнозирование стоимости бриллиантов (Diamonds)
    Основная задача проекта заключается в регрессионном анализе и предсказании рыночной стоимости бриллиантов на основе их физических, геометрических характеристик, а также показателей экспертной оценки качества.
    
    ### Таблица признаков (Features)
    * **carat** (Карат): Вес бриллианта (1 карат = 200 мг). Существенно влияет на цену.
    * **cut** (Огранка): Качество огранки, закодированное числовыми рангами (от плохой к идеальной).
    * **color** (Цвет): Цветовой оттенок бриллианта, представленный числовым кодом.
    * **clarity** (Чистота): Прозрачность камня, отсутствие внутренних и внешних дефектов (числовой код).
    * **depth** (Глубина): Процентное отношение общей глубины камня к его ширине (Формула: z / mean(x, y) = 2 * z / (x + y)).
    * **table** (Площадка): Ширина плоской верхней грани относительно самой широкой части.
    * **x** (Длина): Длина бриллианта в миллиметрах (мм).
    * **y** (Ширина): Ширина бриллианта в миллиметрах (мм).
    * **z** (Высота): Высота бриллианта в миллиметрах (мм).
    * **price** (Цена): Стоимость в долларах США ($) — **Целевая переменная (Target)**.
    
    ### Особенности предобработки данных и EDA:
    1.  **Обработка аномалий:** Была проведена фильтрация нереалистичных физических размеров (значения `x`, `y` или `z` равные 0 были удалены). Аномалии выявлялись с использованием расчетной массы бриллианта.
    2.  **Кодирование категорий:** Ранговые признаки (`cut`, `color`, `clarity`) приведены к числовому формату с помощью Ordinal Encoding.
    3.  **Масштабирование:** Для данных было произведено масштабирование с помощью StandardScaler.
    """)
    
    st.subheader("Интерактивный просмотр структуры обработанных данных")
    df_inf = pd.read_csv("data/diamonds_final.csv")
    st.dataframe(df_inf)
    st.success(f"Cтрок: {df_inf.shape[0]}, столбцов: {df_inf.shape[1]}")

elif page == "Страница 3: Визуализация зависимостей":
    st.title("Разведочный анализ данных (EDA) и графики")
    df = pd.read_csv("data/diamonds_final.csv")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1. Распределение целевой переменной (Price)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(df['price'], kde=False, color='#1f77b4', ax=ax1)
        ax1.set_title("Гистограмма распределения цен на бриллианты")
        ax1.set_xlabel("Цена ($)")
        ax1.set_ylabel("Частота")
        st.pyplot(fig1)
        
        st.markdown("#### 3. Зависимость стоимости от веса (Carat)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x='carat', y='price', alpha=0.5, hue='cut', palette='viridis', ax=ax2)
        ax2.set_title("Влияние веса в каратах на цену бриллианта")
        ax2.set_xlabel("Вес (карат)")
        ax2.set_ylabel("Цена ($)")
        st.pyplot(fig2)

    with col2:
        st.markdown("#### 2. Матрица линейной корреляции")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax3, annot_kws={"size": 8})
        ax3.set_title("Тепловая карта корреляции признаков")
        st.pyplot(fig3)
        
        st.markdown("#### 4. Распределение цен по категориям огранки")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='cut', y='price', palette='pastel', ax=ax4)
        ax4.set_title("Boxplot (распределение цен по категориям огранки)")
        ax4.set_xlabel("Качество огранки (Код)")
        ax4.set_ylabel("Цена ($)")
        st.pyplot(fig4)

elif page == "Страница 4: Инференс моделей ML":
    st.title("Модуль инференса (генерации прогнозов)")
    
    selected_model = st.selectbox("Выберите математическую модель машинного обучения:", list(MODEL_FILES.keys()))
    
    pipeline = load_ml_pipeline(selected_model)
    
    if pipeline is None:
        st.error(f"Файл модели '{MODEL_FILES[selected_model]}' не найден в директории: `{MODEL_DIR}`")
        st.info("Пожалуйста, убедитесь, что файлы моделей сериализованы и лежат по правильному пути.")
    else:
        st.success(f"Пайплайн модели '{selected_model}' успешно инициализирован.")
        
        tab1, tab2 = st.tabs(["Единичный прогноз (Ручной ввод)", "Пакетный прогноз (Загрузка CSV)"])
        
        with tab1:
            st.write("### Укажите параметры бриллианта:")
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                carat = st.number_input("Вес камня в каратах [carat]:", min_value=0.1, max_value=5.0, value=0.7, step=0.01)
                cut = st.slider("Качество огранки [cut] (ранг):", min_value=1, max_value=5, value=4, step=1)
                color = st.slider("Цвет камня [color] (ранг):", min_value=0, max_value=6, value=3, step=1)
                
            with c2:
                clarity = st.slider("Чистота камня [clarity] (ранг):", min_value=1, max_value=8, value=3, step=1)
                depth = st.number_input("Глубина в % [depth]:", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
                table = st.number_input("Ширина площадки в % [table]:", min_value=40.0, max_value=90.0, value=57.0, step=0.1)
                
            with c3:
                x = st.number_input("Длина в мм [x]:", min_value=0.0, max_value=15.0, value=5.5, step=0.01)
                y = st.number_input("Ширина в мм [y]:", min_value=0.0, max_value=15.0, value=5.5, step=0.01)
                z = st.number_input("Высота в мм [z]:", min_value=0.0, max_value=15.0, value=3.4, step=0.01)
            
            valid_inputs = True
            if x <= 0 or y <= 0 or z <= 0:
                st.error("Ошибка валидации: Физические размеры x, y, z должны быть строго больше 0 мм!")
                valid_inputs = False
            elif z >= x or z >= y:
                st.warning("Валидация: Высота (z) обычно меньше длины и ширины бриллианта. Проверьте правильность введенных метрик.")
            
            if st.button("Рассчитать стоимость бриллианта", type="primary", disabled=not valid_inputs):
                input_features = pd.DataFrame([{
                    'carat': carat, 'cut': cut, 'color': color, 'clarity': clarity,
                    'depth': depth, 'table': table, 'x': x, 'y': y, 'z': z
                }])
                
                try:
                    raw_prediction = pipeline.predict(input_features)[0]
                    final_price = max(0.0, float(raw_prediction))
                    
                    st.markdown("---")
                    st.metric(
                        label=f"Оценочная стоимость по модели ({selected_model})",
                        value=f"${final_price:,.2f}"
                    )
                except Exception as e:
                    st.error(f"Ошибка инференса: {e}. Проверьте структуру ожидаемых моделью признаков.")

        with tab2:
            st.write("### Пакетная обработка файлов .csv")
            st.markdown("Загружаемый файл должен содержать заголовки: `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`")
            
            uploaded_batch_file = st.file_uploader("Выберите файл в формате *.csv:", type=["csv"], key="batch_upload")
            
            if uploaded_batch_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_batch_file)
                    required_columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
                    
                    if all(col in batch_df.columns for col in required_columns):
                        features_subset = batch_df[required_columns]
                        
                        batch_predictions = pipeline.predict(features_subset)
                        
                        output_df = batch_df.copy()
                        output_df['predicted_price'] = batch_predictions
                        output_df['predicted_price_formatted'] = output_df['predicted_price'].apply(lambda v: f"${max(0.0, v):,.2f}")
                        
                        st.success("Пакетный инференс успешно выполнен!")
                        st.write("Предварительный просмотр результатов (первые 15 строк):")
                        st.dataframe(output_df.head(15))
                        
                        csv_data = output_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Скачать файл с предсказаниями цены",
                            data=csv_data,
                            file_name=f"predictions_{MODEL_FILES[selected_model].split('.')[0]}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Ошибка валидации файла: В таблице отсутствуют необходимые колонки. Обязательный список: {required_columns}")
                except Exception as e:
                    st.error(f"Произошла техническая ошибка при чтении файла: {e}")
