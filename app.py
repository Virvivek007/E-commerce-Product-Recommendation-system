# -*- coding: utf-8 -*-
# Streamlit App: E-commerce Product Recommendation System

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    data = pd.read_csv("E-commerce-product_dataset.csv")
    data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
    data['Year'] = data['Order Date'].dt.year
    return data

data = load_data()

st.title("📊 E-commerce Sales Forecasting & Product Recommendation System")

# ===============================
# DATA PREVIEW
# ===============================
st.subheader("Dataset Overview")
st.write(data.head())

# ===============================
# DATA VISUALIZATION
# ===============================
st.subheader("Data Visualization")

# 1. Item Type vs Units Sold
st.write("### Item Type vs Units Sold")
fig, ax = plt.subplots()
sns.barplot(y='Item Type', x='Units Sold', data=data, ax=ax)
st.pyplot(fig)

# 2. Profit Trend over Years
st.write("### Total Profit vs Year")
fig, ax = plt.subplots()
sns.lineplot(x='Year', y='Total Profit', data=data, marker='o', ax=ax)
st.pyplot(fig)

# 3. Online vs Offline Pie
st.write("### Online vs Offline Sales Share")
channel_sales = data.groupby(['Sales Channel'])['Units Sold'].sum().reset_index()
fig = px.pie(channel_sales, values="Units Sold", names="Sales Channel", title="Sales Channel Share")
st.plotly_chart(fig)

# 4. Country Heatmap
st.write("### Country-wise Sales Distribution")
country_sales = data.groupby(['Country'])['Units Sold'].sum().reset_index()
fig = px.choropleth(country_sales, locations="Country", locationmode="country names",
                    color="Units Sold", hover_name="Country",
                    title="Country-wise Sales Distribution")
st.plotly_chart(fig)

# ===============================
# POPULARITY-BASED RECOMMENDATION
# ===============================
st.subheader("📦 Popularity-based Recommendation")

region = st.selectbox("Select Region", options=[None] + list(data['Region'].unique()))
channel = st.selectbox("Select Sales Channel", options=[None] + list(data['Sales Channel'].unique()))
top_n = st.slider("Top N Products", 3, 10, 5)

def recommend_popular(region=None, channel=None, top_n=5):
    df = data.copy()
    if region:
        df = df[df['Region'] == region]
    if channel:
        df = df[df['Sales Channel'] == channel]

    top_products = (df.groupby('Item Type')['Units Sold']
                      .sum()
                      .sort_values(ascending=False)
                      .head(top_n))
    return top_products

if st.button("Recommend Popular Products"):
    st.write(recommend_popular(region, channel, top_n))

# ===============================
# CONTENT-BASED RECOMMENDATION
# ===============================
st.subheader("🧠 Content-based Recommendation (Cosine Similarity)")

features = ['Item Type', 'Region', 'Country', 'Sales Channel']
encoded_data = data[features].apply(LabelEncoder().fit_transform)
similarity = cosine_similarity(encoded_data)

item_index = st.number_input("Enter Item Index (row number)", min_value=0, max_value=len(data)-1, value=0)

def recommend_similar(item_index, top_n=5):
    sim_scores = list(enumerate(similarity[item_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_items = data.iloc[[i[0] for i in sim_scores]][['Item Type','Region','Country','Sales Channel']]
    return recommended_items

if st.button("Find Similar Products"):
    st.write(recommend_similar(item_index))

# ===============================
# FORECASTING MODEL (SARIMA)
# ===============================
st.subheader("📈 Sales Forecasting")

product = st.selectbox("Select Item Type for Forecast", options=data['Item Type'].unique())

trend_data = data.groupby(['Year', 'Item Type'])['Units Sold'].sum().reset_index()
product_sales = trend_data[trend_data['Item Type'] == product][['Year', 'Units Sold']]
product_sales = product_sales.set_index('Year')
if len(product_sales) > 3:
    try:
        model = SARIMAX(product_sales, order=(1,1,1), seasonal_order=(1,1,1,3))
        results = model.fit(disp=False)

        forecast = results.get_forecast(steps=5)
        forecast_values = pd.Series(np.ravel(forecast.predicted_mean))
        forecast_index = list(range(product_sales.index.max()+1, product_sales.index.max()+6))

        fig, ax = plt.subplots()
        ax.plot(product_sales.index, product_sales['Units Sold'], marker='o', label="Actual Sales")

        if len(forecast_values) == len(forecast_index):
            ax.plot(forecast_index, forecast_values, marker='x', linestyle='--', label="Forecast")

        ax.set_title(f"Sales Forecast for {product}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Units Sold")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Model failed: {e}")
else:
    st.warning("Not enough data points for forecasting.")
