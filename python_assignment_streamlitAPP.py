import streamlit as st
import requests
from datetime import datetime
import plotly.graph_objs as go
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from PIL import Image, ImageOps
import numpy as np
from keras.preprocessing.image import img_to_array

# 将时间戳转换为年月日形式
def format_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')

# 将时间戳转换为年月日形式
def format_timestamp_ms(timestamp):
    dt_object = datetime.fromtimestamp(timestamp / 1000)
    formatted_date = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_date

# 获取所有的coin代码，以便后续函数可以获取coin_id
def get_all_coins():
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return []

# 获取所有货币的id 
def fetch_crypto_prices_for_all_coins():
    coins = get_all_coins()
    prices = []
    for coin in coins:
        coin_id = coin['id']
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=cad&days=365&precision=4"
        response = requests.get(url)
        if response.status_code == 200:
            # 处理响应数据，将其添加到 prices 列表中
            prices.append(response.json())
    return prices

# 根据coin_id获取过去一年的价格
def fetch_market_chart_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=cad&days=365&precision=4"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
# 根据coin_id和天数获取过去1week, 1month, 1year, 5years的价格
def fetch_market_chart_data_withDay(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=cad&days={days}&precision=4"
    response = requests.get(url)
    data = response.json()

    # 检查返回内容并作出相应处理
    if 'status' in data and 'error_code' in data['status']:
        error_code = data['status']['error_code']
        if error_code == 429:
            st.write("You've exceeded the Rate Limit. Please wait for a few minutes to compare.")
            return None, None

    if 'error' in data and 'status' in data['error']:
        error_status = data['error']['status']
        if error_status['error_code'] == 10012:
            st.write("We don't provide 5 years data since the API request exceeds the allowed time range.")
            return None, None
        
    if response.status_code == 200:
        response_raw = response.json()
        response_price = response_raw['prices']
        timestamps = []
        prices = []
        for timestamp, price in response_price:
            formatted_date = format_timestamp_ms(timestamp)
            timestamps.append(formatted_date)
            prices.append(price)
        #return response.json()
        return timestamps,prices
    else:
        return None, None
    
# 解析上传的照片对应的数字
def identify_number(uploaded_file):
    model = tf.keras.models.load_model('numberclassifier.keras')
    model_summary = model.summary()
    #st.write(model_summary)
    image = Image.open(uploaded_file)

    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        channels = image.split()
        if len(channels) == 4:
            alpha = channels[-1]
            bg = Image.new("RGB", image.size, (255,255,255))
            bg.paste(image, mask=alpha)
            image = bg

    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)

    image = image.resize((28,28), Image.Resampling.LANCZOS)
    image_array = img_to_array(image)/255.0
    image_array = np.expand_dims(image_array,axis=0)
    prediction = model.predict(image_array)
    prediction = tf.nn.softmax(prediction).numpy()

    return np.argmax(prediction)


# 根据coin_symbol 返回coin_id
def get_coin_symbol_id(coin_symbol):
    coin_id = None
    all_coins = get_all_coins()
    coin_found = False
    for coin in all_coins:
        if coin_symbol.lower() == coin['name'].lower():
            coin_id = coin['id']
            coin_found = True
            break
    if not coin_found:
        st.write(coin_symbol + " is not correct. "+ "Please input correct coin symbol!")
    return coin_id

#######################################################################################################
# 展示界面1   
def show_stock_details_page():
    st.title('Stock Details')
    all_coins = get_all_coins() #获取所有的货币信息，包含symbol，id

    # 添加输入框，用于输入coin symbol
    coin_symbol = st.text_input("Enter coin symbol:")
    
    if all_coins:
        # 检查输入的coin symbol是否存在于coin['name']中，存在的话，则获取对应的coin_id
        if coin_symbol:
            coin_id = None
            for coin in all_coins:
                if coin_symbol.lower() == coin['name'].lower():
                    coin_id = coin['id']
                    break
            
            if coin_id:
                st.write(f"The ID for {coin_symbol} is: {coin_id}")
                # 调用函数获取市场价格信息
                market_chart_data = fetch_market_chart_data(coin_id)
                if market_chart_data:
                    # 处理市场价格信息
                    st.write("Market Chart Data is as below:")
                    timestamps = []
                    prices = []
                    for price in market_chart_data['prices']:
                        timestamp, price_value = price
                        formatted_date = format_timestamp(timestamp)
                        timestamps.append(formatted_date)
                        prices.append(price_value)

                    # 使用 Plotly 绘制折线图
                    fig = go.Figure(data=go.Scatter(x=timestamps, y=prices, mode='lines', name='Price'))
                    fig.update_layout(title=f'Price Chart for {coin_symbol}', xaxis_title='Date', yaxis_title='Price (CAD)')
                    st.plotly_chart(fig)

                    # 获取最高价格和对应的日期
                    max_price = max(prices)
                    max_price_index = prices.index(max_price)
                    max_price_date = timestamps[max_price_index]
                    st.write(f"Highest Price: {max_price} (on {max_price_date})")

                    # 获取最低价格和对应的日期
                    min_price = min(prices)
                    min_price_index = prices.index(min_price)
                    min_price_date = timestamps[min_price_index]
                    st.write(f"Lowest Price: {min_price} (on {min_price_date})")

                else:
                    st.write("Failed to fetch market chart data")
            else:
                st.write("Please input correct cryptocurrency name")
    else:
        st.write("Failed to fetch coin data")

######################################################################################################

# 第二个界面
def show_coin_comparison_page():
    st.title('Coin Comparison')
    st.write("Enter the symbols of the two cryptocurrencies to compare:")

    # 输入框，用于输入两个coin symbol
    coin_symbol1 = st.text_input("Enter the first coin symbol:")
    coin_symbol2 = st.text_input("Enter the second coin symbol:")

    # 调用函数获取市场价格信息
    time_range = st.selectbox("Select time range:", ["1 week", "1 month", "1 year", "5 years"])
    
    # 获取coin_symbol1和coin_symbol2对应的coin_id1和coin_id2
    if st.button("Compare"):
        
        #设置coin_symbol1和coin_symbol2是必填项
        if not coin_symbol1 or not coin_symbol2:
            st.write("Please input the coin symbols, they are mandatory!")
            return

        if time_range == "1 week":
            days = 7
        elif time_range == "1 month":
            days = 30
        elif time_range == "1 year":
            days = 365
        elif time_range == "5 years":
            days = 1825

        coin_id1 = get_coin_symbol_id(coin_symbol1)
        coin_id2 = get_coin_symbol_id(coin_symbol2)

        market_datetimes1, market_prices1 = fetch_market_chart_data_withDay(coin_id1,days)

        market_datetimes2, market_prices2 = fetch_market_chart_data_withDay(coin_id2,days)

        # 使用 Plotly 绘制折线图，并添加第二条折线
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=market_datetimes1, y=market_prices1, mode='lines', name=coin_symbol1))
        fig.add_trace(go.Scatter(x=market_datetimes2, y=market_prices2, mode='lines', name=coin_symbol2))

        fig.update_layout(title=f'Price Chart for Coin Comparison', xaxis_title='Date', yaxis_title='Price (CAD)')
        st.plotly_chart(fig)

######################################################################################################

# 第三个界面
def show_image_classifier_page():
    st.title('Image Classifier')
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.info("Image Identified as " + str(identify_number(uploaded_file)))
        st.image(uploaded_file, caption='Uploaded Image.', width=100)

    # Add your code for Image Classifier page here



selected_page = st.sidebar.radio('Navigation', ['Stock Details', 'Coin Comparison', 'Image Classifier'])

# Display the content based on the page selected by user
if selected_page == 'Stock Details':
    show_stock_details_page()
elif selected_page == 'Coin Comparison':
    show_coin_comparison_page()
elif selected_page == 'Image Classifier':
    show_image_classifier_page()
