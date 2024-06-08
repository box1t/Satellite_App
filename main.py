import streamlit as st
import streamlit_authenticator as stauth
import numpy as np
import cv2 as cv
import os
from streamlit import session_state
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import get_custom_objects
from keras.preprocessing.image import load_img
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, BatchNormalization, concatenate
import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont

import sqlite3 

import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import pandas as pd

import datetime

import hashlib
import re

conn = sqlite3.connect('data.db')
c = conn.cursor()

def validate_password(password):
    if re.search('[а-яА-ЯёЁ]', password) is not None: return False
    if len(password) < 8: return False
    if re.search('[A-Z]', password) is None: return False
    return True

def validate_username(username):
    if len(username) < 4: return False
    if re.search('[0-9]', username): return False
    if re.search('[а-яА-ЯёЁ]', username) is not None: return False
    return True

def validate(username, password):
    if validate_password(password) and validate_username(username): return True

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username,password))
	data = c.fetchall()
	return data

def logout_user(username, password):
    c.execute('DELETE FROM userstable WHERE username =? AND password = ?', (username,password))

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

def create_result_table():
    c.execute('CREATE TABLE IF NOT EXISTS resulttable(username TEXT,date TEXT,ioumetric INTEGER)')

def insert_iou_metric(username, date, ioumetric):
    create_result_table()
    c.execute('INSERT INTO resulttable(username,date,ioumetric) VALUES (?,?,?)',(username, date,ioumetric))
    conn.commit()

def get_iou_metric(username,date, ioumetric):
    create_result_table()
    c.execute('SELECT * FROM resulttable WHERE username =? AND date =? AND ioumetric = ?', (username,date,ioumetric))
    data = c.fetchall()
    return data

def get_all_iou_metric():
    create_result_table()
    c.execute('SELECT * FROM resulttable')
    data = c.fetchall()
    return data


# Initialize the session state variables
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = 'Войти в аккаунт'

if st.session_state.authenticated:
    st.sidebar.write(f"Имя пользователя: {st.session_state.username}")
    st.session_state.page = st.sidebar.selectbox('Меню', ['О нас', 'Анализ спутниковых снимков', 'Прошлые результаты'])
    if st.sidebar.button('Выйти из аккаунта', key='logout_sidebar'):
        logout_user('Выйти из аккаунта', 'sidebar')
        st.success('Вы вышли из аккаунта')
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.page = 'Войти в аккаунт'
        st.experimental_rerun()
else:
    st.session_state.page = st.sidebar.selectbox('Меню', ['О нас', 'Войти в аккаунт', 'Зарегистрироваться', 'Анализ спутниковых снимков', 'Прошлые результаты'])

page = st.session_state.page

def display_summary_statistics(history_dict, model_name):
    """
    Display the summary statistics for the given model in the Streamlit app.
    """
    summary_statistics = {}
    st.write(f"**Обобщенные метрики для модели {model_name}:**")
    for key, values in history_dict.items():
        mean = np.mean(values)
        std = np.std(values)
        max_val = np.max(values)
        summary_statistics[key] = values
        st.write(f"{key}: ")
        st.write(f"- Среднее значение = {mean:.2f}")
        st.write(f"- Среднеквадратичное отклонение = {std:.2f}")
        st.write(f"- Максимальное значение = {max_val:.2f}")
        donut_chart = make_donut(round(mean*100, 2), 'Среднее значение', 'blue')
        st.altair_chart(donut_chart, use_container_width=True)

    return summary_statistics

def plot_correlations(history_dict, model_name):
    """
    Plot the correlations between the loss and accuracy for the given model.
    """
    # Training Loss vs. Training Accuracy
    x_coords = history_dict['train_loss']
    y_coords = history_dict['train_acc']

    z = np.polyfit(x_coords, y_coords, 1)  # Calculate the slope and intercept of the trend line
    p = np.poly1d(z)  # Create a polynomial function that represents the trend line

    fig_1 = go.Figure(
        data=[
            go.Scatter(  # Modify the name of the scatter plot trace
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=12,
                    color='blue',
                    line=dict(
                        width=2,
                        color='darkblue'
                    )
                ),
                name='Обучающая выборка'
            ),
            go.Scatter(  # Modify the name of the trend line trace
                x=x_coords,
                y=p(x_coords),
                mode='lines',
                line=dict(
                    color='red',
                    width=4,
                    dash='dash'
                ),
                name='Линия тренда'
            )
        ],
        layout=go.Layout(
            title=dict(
                text=f'Погрешность и точность при обучении для {model_name}',
                xref='paper',
                x=0.05
            ),
            xaxis=dict(
                title='Погрешность при обучении',
                zeroline=False,
                gridcolor='lightgray',
                gridwidth=2
            ),
            yaxis=dict(
                title='Точность обучения',
                zeroline=False,
                gridcolor='lightgray',
                gridwidth=2
            ),
            plot_bgcolor='white',
            width=800,
            height=600
        )
    )

    # Validation Loss vs. Validation Accuracy
    x_coords = history_dict['valid_loss']
    y_coords = history_dict['valid_acc']

    z = np.polyfit(x_coords, y_coords, 1)  # Calculate the slope and intercept of the trend line
    p = np.poly1d(z)  # Create a polynomial function that represents the trend line

    fig_2 = go.Figure(
        data=[
            go.Scatter(  # Modify the name of the scatter plot trace
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=12,
                    color='green',
                    line=dict(
                        width=2,
                        color='darkgreen'
                    )
                ),
                name='Контрольная выборка'
            ),
            go.Scatter(  # Modify the name of the trend line trace
                x=x_coords,
                y=p(x_coords),
                mode='lines',
                line=dict(
                    color='red',
                    width=4,
                    dash='dash'
                ),
                name='Линия тренда'
            )
        ],
        layout=go.Layout(
            title=dict(
                text=f'Контрольные погрешность и точность для {model_name}',
                xref='paper',
                x=0.05
            ),
            xaxis=dict(
                title='Контрольная погрешность',
                zeroline=False,
                gridcolor='lightgray',
                gridwidth=2
            ),
            yaxis=dict(
                title='Контрольная точность',
                zeroline=False,
                gridcolor='lightgray',
                gridwidth=2
            ),
            plot_bgcolor='white',
            width=800,
            height=600
        )
    )

    # Display the plots using Streamlit
    st.plotly_chart(fig_1)
    st.plotly_chart(fig_2)

def plot_learning_curves(history_dict, model_name):
    """
    Plot the learning curves for the given model.
    """
    epochs = list(range(1, len(history_dict['train_loss']) + 1))  # Convert to list

    # Loss
    fig_1 = go.Figure(
        data=[
            go.Scatter(
                x=epochs,  # Use the list
                y=history_dict['train_loss'],
                name='Погрешность при обучении',
                line=dict(
                    color='blue',
                    width=4
                )
            ),
            go.Scatter(
                x=epochs,  # Use the list
                y=history_dict['valid_loss'],
                name='Контрольная погрешность',
                line=dict(
                    color='green',
                    width=4
                )
            )
        ],
        layout=go.Layout(
            title=dict(
                text=f'Погрешность для {model_name}',
                xref='paper',
                x=0.05
            ),
            xaxis=dict(
                title='Число эпох',
                zeroline=False,
                gridcolor='lightgray',
                gridwidth=2
            ),
            yaxis=dict(
                title='Погрешность',
                zeroline=False,
                gridcolor='lightgray',
                gridwidth=2
            ),
            plot_bgcolor='white',
            width=800,
            height=600
        )
    )

    # Accuracy
    fig_2 = go.Figure(
        data=[
            go.Scatter(
                x=epochs,  # Use the list
                y=history_dict['train_acc'],
                name='Точность обучения',
                line=dict(
                    color='blue',
                    width=4
                )
            ),
            go.Scatter(
                x=epochs,  # Use the list
                y=history_dict['valid_acc'],
                name='Контрольная точность',
                line=dict(
                    color='green',
                    width=4
                )
            )
        ],
        layout=go.Layout(
            title=dict(
                text=f'Точность для {model_name}',
                xref='paper',
                x=0.05
            ),
            xaxis=dict(
                title='Число эпох',
                zeroline=False,
                gridcolor='lightgray',
                gridwidth=2
            ),
            yaxis=dict(
                title='Точность',
                zeroline=False,
                gridcolor='lightgray',
                gridwidth=2
            ),
            plot_bgcolor='white',
            width=800,
            height=600
        )
    )

    # Display the plots using Streamlit
    st.plotly_chart(fig_1)
    st.plotly_chart(fig_2)

def make_donut(input_response, input_text, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100-input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    inner_radius = 35
    corner_radius = inner_radius / 3

    plot = alt.Chart(source).mark_arc(innerRadius=inner_radius, cornerRadius=corner_radius).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                          scale=alt.Scale(
                              domain=[input_text, ''],
                              range=chart_color),
                          legend=None),
        ).properties(width=100, height=100)

    text = plot.mark_text(align='center', color="#29b5e8", font="Arial Black", fontSize=15, fontWeight=700, dx=0).encode(text=alt.value(f'{input_response}%'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=inner_radius, cornerRadius=corner_radius).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                          scale=alt.Scale(
                              domain=[input_text, ''],
                              range=chart_color),  # 31333F
                          legend=None),
    ).properties(width=100, height=100)
    return plot_bg + plot + text


def create_pdf_report(username, summary_statistics):
    """
    Create a PDF report with the given page, username, summary statistics, and plots.
    """
    # Path to the font file
    font_path = "/home/snowwy/Desktop/DZZ/diff/DejaVuSans.ttf"  # Update this to the actual path

    # Register the font
    pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))

    # Get the sample style sheet and modify existing styles or add new ones
    styles = getSampleStyleSheet()
    
    # Modify the existing 'Title' style
    styles['Title'].fontName = 'DejaVuSans'
    styles['Title'].fontSize = 18
    styles['Title'].leading = 22

    # Add new styles with unique names
    styles.add(ParagraphStyle(name='MyBodyText', fontName='DejaVuSans', fontSize=12, leading=14))
    styles.add(ParagraphStyle(name='MyHeading2', fontName='DejaVuSans', fontSize=14, leading=16))

    doc = SimpleDocTemplate("/home/snowwy/Desktop/DZZ/diff/report2.pdf", pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    elements = []

    # Add the page title
    elements.append(Paragraph("Результаты сегментации паводков", styles["Title"]))

    # Add a space
    elements.append(Spacer(1, 12))

    # Add the username
    elements.append(Paragraph(f"Имя пользователя: {username}", styles["MyBodyText"]))

    # Add a space
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Итоговая статистика:", styles["MyHeading2"]))
    for key, values in summary_statistics.items():
        mean = np.mean(values)
        std = np.std(values)
        max_val = np.max(values)
        elements.append(Paragraph(f"{key}:", styles["MyBodyText"]))
        elements.append(Paragraph(f"- Среднее значение = {mean:.2f}", styles["MyBodyText"]))
        elements.append(Paragraph(f"- Среднеквадратичное отклонение = {std:.2f}", styles["MyBodyText"]))
        elements.append(Paragraph(f"- Максимальное значение = {max_val:.2f}", styles["MyBodyText"]))

        # Generate the donut chart and save it as an image file
        # donut_chart_path = make_donut(round(mean*100, 2), 'Mean', 'blue')

        # Add the image file to the PDF
        # elements.append(Image(donut_chart_path, width=200, height=200))

    # Build the document
    doc.build(elements)

if page == 'О нас':
    html_code = open("faq.html", 'r', encoding='utf-8').read()
    st.markdown(html_code, unsafe_allow_html=True)


elif page == 'Войти в аккаунт':
    st.title("Вход в аккаунт")
    user = st.text_input('Имя пользователя')
    passwd = st.text_input('Пароль', type='password')
    
    if st.button('Войти') :
        create_usertable()
        hashed_password = make_hashes(passwd)
        result = login_user(user, check_hashes(passwd, hashed_password))
        st.session_state.username = user
        if result:
            st.success(f'Вы вошли как: {st.session_state.username}')
            st.session_state.authenticated = True
            st.session_state.page = 'Анализ спутниковых снимков' 
            st.rerun()
        else:
            st.error('Логин или пароль введены некорректно')
            st.info("Нет аккаунта? Зарегистрируйтесь")

elif page == "Зарегистрироваться":
    st.title("Регистрация")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type='password')

    if (st.button("Зарегистрироваться")):
        create_usertable()
        if validate_username(new_user) and validate_password(new_password):
            result = login_user(new_user, make_hashes(new_password))
            if result:
                st.error("Аккаунт с таким именем и паролем уже существует")
            else:
                add_userdata(new_user, make_hashes(new_password))
                st.success("Спасибо за регистрацию! Аккаунт был создан")
                login_user(new_user, make_hashes(new_password))
                st.session_state.username = new_user
                st.session_state.authenticated = True
                st.info("Автоматический вход в аккаунт выполнен")
                st.session_state.page = 'Анализ спутниковых снимков' 
                st.rerun()
        else:
            if validate_username(new_user) == False:
                st.error('Некорретное имя пользователя')
                st.info('Проверьте, что имя содержит только символы латинских букв и его длина больше 4 символов')
            if validate_password(new_password) == False:
                st.error('Некорретный пароль')
                st.info('Проверьте, что пароль содержит только символы латинских букв, цифры, хотя бы одну заглавную букву и его длина больше 8 символов')
            
elif page == "Прошлые результаты":
    st.title("Прошлые результаты")
    res = get_all_iou_metric()
    for item in res:
        st.subheader("имя пользователя: " + item[0])
        st.text("дата: " + item[1])
        st.text("iou метрика: " + str(item[2]))

elif page == 'Анализ спутниковых снимков':
    if not st.session_state.get('authenticated', False):
        st.warning('Для доступа к приложению необходимо войти в аккаунт')
        st.stop()
    elif st.session_state.authenticated and page == 'Войти в аккаунт':
        st.warning('Вы уже вошли в аккаунт')
        st.stop()

    # Image Segmentation App
    st.title("Сегментация паводков на спутниковых снимках")
    st.info("Вы можете выбрать модель сегментацию с помощью бокового меню")
    st.sidebar.title("Анализ данных")
    model_choice = st.sidebar.selectbox("Выберите модель сегментации:", ["UNet", "Attention UNet"])


    if st.button('Выгрузить отчет в PDF'):
        
        if model_choice == "UNet":
            history_dict = np.load('unet_history.npy', allow_pickle=True).item()
            model_name = "UNet"
        elif model_choice == "Attention UNet":
            history_dict = np.load('attention_history.npy', allow_pickle=True).item()
            model_name = "Attention Unet"
        
        summary_statistics = display_summary_statistics(history_dict, model_name)

        # Create the PDF report
        create_pdf_report(st.session_state.username, summary_statistics)


    class EncoderBlock(tf.keras.layers.Layer):
        def __init__(self, filters, rate, pooling=True, **kwargs):
            super(EncoderBlock, self).__init__(**kwargs)
            self.filters = filters
            self.rate = rate
            self.pooling = pooling
            self.conv1 = Conv2D(self.filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
            self.conv2 = Conv2D(self.filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
            self.pool = MaxPool2D(pool_size=(2, 2))

        def call(self, inputs):
            x = self.conv1(inputs)
            x = self.conv2(x)
            if self.pooling:
                y = self.pool(x)
                return y, x
            else:
                return x

        def get_config(self):
            base_config = super().get_config()
            return {
                **base_config,
                "filters": self.filters,
                "rate": self.rate,
                "pooling": self.pooling
            }

    class DecoderBlock(tf.keras.layers.Layer):
        def __init__(self, filters, rate, **kwargs):
            super(DecoderBlock, self).__init__(**kwargs)
            self.filters = filters
            self.rate = rate
            self.up = UpSampling2D()
            self.net = EncoderBlock(filters, rate, pooling=False)

        def call(self, inputs):
            X, short_X = inputs
            ct = self.up(X)
            c_ = concatenate([ct, short_X])
            x = self.net(c_)
            return x

        def get_config(self):
            base_config = super().get_config()
            return {
                **base_config,
                "filters": self.filters,
                "rate": self.rate
            }

    class AttentionChannel(tf.keras.layers.Layer):
        def __init__(self, filters, **kwargs):
            super(AttentionChannel, self).__init__(**kwargs)
            self.filters = filters
            self.C1 = Conv2D(filters, kernel_size=1, strides=1, padding='same', activation=None)
            self.C2 = Conv2D(filters, kernel_size=1, strides=2, padding='same', activation=None)
            self.relu = tf.keras.activations.relu
            self.add = tf.keras.layers.Add()
            self.C3 = Conv2D(1, kernel_size=1, strides=1, padding='same', activation='sigmoid')
            self.up = tf.keras.layers.UpSampling2D()
            self.mul = tf.keras.layers.Multiply()
            self.BN = BatchNormalization()

        def call(self, X):
            org_x, skip_g = X
            g = self.C1(org_x)
            x = self.C2(skip_g)
            x = self.add([g, x])
            x = self.C3(x)
            x = self.up(x)
            x = self.mul([x, skip_g])
            x = self.BN(x)
            return x

        def get_config(self):
            base_config = super().get_config()
            base_config.update({"filters": self.filters})
            return base_config

    get_custom_objects().update({
        'EncoderBlock': EncoderBlock,
        'DecoderBlock': DecoderBlock,
        'AttentionChannel': AttentionChannel
    })

    @st.cache_resource
    def load_unet_model():
        with tf.keras.utils.custom_object_scope({'EncoderBlock': EncoderBlock, 'DecoderBlock': DecoderBlock}):
            return load_model('UNet.h5', compile=False)

    @st.cache_resource
    def load_attention_unet_model():
        with tf.keras.utils.custom_object_scope({'EncoderBlock': EncoderBlock, 'DecoderBlock': DecoderBlock, 'AttentionChannel': AttentionChannel}):
            return load_model('AttentionUNet.h5', compile=False)

    def preprocess_image(image, size=128):
        image = cv.resize(image, (size, size), interpolation=cv.INTER_AREA)
        image = image.astype('float32') / 255.
        image = np.expand_dims(image, axis=0)
        return image

    def postprocess_mask(mask, target_size):
        mask = mask.reshape(128, 128)
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = cv.resize(mask, target_size, interpolation=cv.INTER_NEAREST)
        return mask

    def show_image(image, title=None, cmap=None):
        plt.figure()
        plt.title(title)
        plt.imshow(image, cmap=cmap)
        plt.axis('off')

    try:
        model = load_unet_model() if model_choice == "UNet" else load_attention_unet_model()
    except FileNotFoundError as e:
        st.error(e)
        st.stop()

    uploaded_file = st.file_uploader("Загрузите изображение в JPEG-формате...", type="jpeg")

    output_dir = 'generated_masks'
    os.makedirs(output_dir, exist_ok=True)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, 1)
        input_size = (img.shape[1], img.shape[0])
        img_resized = preprocess_image(img)

        mask_pred = model.predict(img_resized)
        mask_post = postprocess_mask(mask_pred[0], input_size)

        input_filename = uploaded_file.name
        output_path = os.path.join(output_dir, input_filename)
        cv.imwrite(output_path, mask_post)

        col1, col2 = st.columns(2)

        with col1:
            st.header("Загруженное изображение")
            st.image(img, channels="BGR")

        with col2:
            st.header("Предсказанная маска")
            st.image(mask_post, channels="GRAY")

        st.header("Анализируем результаты:")
        iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)
        ground_truth_dir = 'generated_masks'
        ground_truth_path = os.path.join(ground_truth_dir, input_filename)

        mask_true = preprocess_image(cv.imread(ground_truth_path, cv.IMREAD_GRAYSCALE)).reshape(128, 128, 1)
        iou_metric.update_state(mask_true, mask_pred[0])
        iou_score = iou_metric.result().numpy()

        data = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        insert_iou_metric(st.session_state.username, data, round(iou_score * 100, 2))

        donut_chart = make_donut(round(iou_score*100, 2), 'Индекс Жаккарда', 'blue')
        st.altair_chart(donut_chart, use_container_width=True)

        fig, ax = plt.subplots(figsize=(15, 5))

        def plot_history(history, model_name, ax):
            ax.plot(history['train_acc'], label=f'{model_name} - точность обучения')
            ax.plot(history['valid_acc'], label=f'{model_name} - контрольная точность')
            ax.plot(history['train_loss'], label=f'{model_name} - погрешность обучения')
            ax.plot(history['valid_loss'], label=f'{model_name} - контрольная погрешность')
            ax.plot(history['train_IoU'], label=f'{model_name} - индекс Жаккарда при обучении')
            ax.plot(history['valid_IoU'], label=f'{model_name} - контрольный индекс Жаккарда')
            ax.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
            ax.set_title(f'Кривые обучения модели {model_name}')
            ax.grid()

        if model_choice == "UNet":
            unet_history = np.load('unet_history.npy', allow_pickle=True).item()
            plot_history(unet_history, "UNet", ax)
            display_summary_statistics(unet_history, "UNet")
            st.header("Кривые обучения модели")
            plot_correlations(unet_history, "UNet")
            plot_learning_curves(unet_history, "UNet")
        elif model_choice == "Attention UNet":
            attention_history = np.load('training_history/attention_history.npy', allow_pickle=True).item()
            plot_history(attention_history, "Attention UNet", ax)
            display_summary_statistics(attention_history, "Attention UNet")
            st.header("Кривые обучения для модели")
            plot_correlations(attention_history, "Attention UNet")
            plot_learning_curves(attention_history, "Attention UNet")
        st.pyplot(fig)
