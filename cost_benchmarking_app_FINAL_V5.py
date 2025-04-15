# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:11:32 2025

@author: bansala4846
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error
import io
import base64
from datetime import datetime
import logging
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ------------------ 🌐 APP CONFIG ------------------
st.set_page_config(
    layout="wide",
    page_title="Arcadis Cost Intelligence Suite",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

import time

# --- Enhanced First Load Animation ---
if "loaded" not in st.session_state:
    with st.spinner("🔄 Loading Arcadis Cost Intelligence Suite..."):
        loading_msg = st.empty()
        progress_bar = st.progress(0)
        for percent in range(101):
            time.sleep(0.015)
            progress_bar.progress(percent)
            loading_msg.text(f"Loading... {percent}%")
        time.sleep(0.3)
        loading_msg.empty()
    st.session_state.loaded = True

# ------------------ 🌐 SIDEBAR PANEL ------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/10/Arcadis_Logo.svg", width=150)
    st.markdown("<h2 style='color: #1E88E5;'>Arcadis Control Panel</h2>", unsafe_allow_html=True)
    dark_mode = st.checkbox("Dark Mode")
    if dark_mode:
        st.markdown('<style>.main {background-color: #263238; color: #FFFFFF;}</style>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload Your Dataset", type=["xlsx"])

    # AI Assistant
    st.subheader("🤖 Arcadis AI Assistant")
    question = st.text_input("Ask about your data")
    if st.button("Ask"):
        st.write(f"AI Response: '{question}' analysis in progress... (Full AI integration soon!)")

# ------------------ 🚀 LANDING PAGE ------------------
if not uploaded_file:
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h1 style='color:#1E88E5;'>🚀 Arcadis Cost Intelligence Suite</h1>
            <p style='font-size:18px; color: #546E7A;'>Empower your decisions with Arcadis’ cutting-edge cost analytics platform.</p>
            <img src='https://media.giphy.com/media/l0ExvXIe8jJcrXbW0/giphy.gif' width='300'>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ------------------ 📊 DASHBOARD TABS ------------------

@st.cache_data
def load_data(file_content):
    return pd.read_excel(io.BytesIO(file_content))

@st.cache_data
def filter_data(data, sector_filter, asset_filter, cost_range, search_term):
    filtered = data[
        (data['Sector'].isin(sector_filter)) &
        (data['Asset Type'].isin(asset_filter)) &
        (data['Cost (£m)'].between(cost_range[0], cost_range[1]))
    ]
    if search_term:
        filtered = filtered[filtered.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)]
    return filtered

@st.cache_data
def forecast_cost(df):
    required = ['Duration (months)', 'Manual Handling %', 'Quality Score', 'Cost (£m)']
    if not all(col in df.columns for col in required):
        return df, 0.0
    X = df[required[:-1]].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.to_numeric(df['Cost (£m)'], errors='coerce').fillna(0)
    if len(X) < 2:
        return df, 0.0
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    predicted = model.predict(X)
    mse = mean_squared_error(y, predicted)
    return pd.concat([df, pd.Series(predicted, name='Predicted Cost', index=df.index)], axis=1), mse

@st.cache_data
def detect_anomalies(df):
    if not all(col in df.columns for col in ['Cost (£m)', 'Duration (months)']):
        return df
    X = df[['Cost (£m)', 'Duration (months)']].apply(pd.to_numeric, errors='coerce').fillna(0)
    iso = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly'] = iso.fit_predict(X) == -1
    return df

def generate_ai_insight(df):
    avg_cost = df['Cost (£m)'].mean()
    avg_overrun = df['Forecasted Overrun (%)'].mean()
    return f"Arcadis AI Insight: Portfolio (£{avg_cost:,.2f}m avg cost, {avg_overrun:.1f}% overrun) suggests {'cost optimization' if avg_cost > 50 else 'process review' if avg_overrun > 10 else 'strong performance'}."

def generate_executive_summary(df):
    return f"Executive Summary: Your portfolio of {len(df)} projects, valued at £{df['Cost (£m)'].sum():,.2f}m, shows an average overrun of {df['Forecasted Overrun (%)'].mean():.1f}%. Arcadis recommends focusing on {'cost efficiency' if df['Cost (£m)'].mean() > 50 else 'schedule adherence'}."

def get_download_link(df, filename="arcadis_benchmarking_output.xlsx"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    b64 = base64.b64encode(output.getvalue()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download Excel</a>'

def generate_pdf_report(df, filename="arcadis_benchmarking_report.pdf"):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Arcadis Cost Intelligence Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(100, 710, f"Projects: {len(df)} | Value: £{df['Cost (£m)'].sum():,.2f}m")
    c.showPage()
    c.save()
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📜 Download PDF Report</a>'

# Load data
raw_data = load_data(uploaded_file.getvalue())
sector_filter = st.sidebar.multiselect("Sectors", sorted(raw_data['Sector'].unique()), default=list(raw_data['Sector'].unique()))
asset_filter = st.sidebar.multiselect("Asset Types", sorted(raw_data['Asset Type'].unique()), default=list(raw_data['Asset Type'].unique()))
cost_range = st.sidebar.slider("Cost Range (£m)", float(raw_data['Cost (£m)'].min()), float(raw_data['Cost (£m)'].max()), (float(raw_data['Cost (£m)'].min()), float(raw_data['Cost (£m)'].max())))
search_term = st.sidebar.text_input("Search Projects")
filtered_data = filter_data(raw_data, sector_filter, asset_filter, cost_range, search_term)

tabs = st.tabs(["🌟 Overview", "📈 Executive Insights", "🔍 Deep Dive", "🔮 Predictive Analytics"])  # 🌐 Navigation Tabs

# ------------------ 🌟 OVERVIEW TAB ------------------
with tabs[0]:
    st.header("🌟 Portfolio Snapshot")
    st.markdown(f"<p style='color: #FF7043;'>{generate_ai_insight(filtered_data)}</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Projects", f"{len(filtered_data):,}")
    col2.metric("Value", f"£{filtered_data['Cost (£m)'].sum():,.2f}m")
    col3.metric("Avg Overrun", f"{filtered_data['Forecasted Overrun (%)'].mean():.1f}%")

    st.markdown(get_download_link(filtered_data), unsafe_allow_html=True)
    st.markdown(generate_pdf_report(filtered_data), unsafe_allow_html=True)

# ------------------ 📈 EXECUTIVE INSIGHTS ------------------
with tabs[1]:
    st.header("📈 Executive Insights")
    st.write(generate_executive_summary(filtered_data))

# ------------------ 🔍 DEEP DIVE ------------------
with tabs[2]:
    st.header("🔍 Deep Dive Analytics")
    fig = px.scatter_3d(filtered_data, x='Duration (months)', y='Cost (£m)', z='Forecasted Overrun (%)', color='Sector')
    st.plotly_chart(fig, use_container_width=True)

# ------------------ 🔮 PREDICTIVE ------------------
with tabs[3]:
    st.header("🔮 Predictive Analytics")
    forecasted, mse = forecast_cost(filtered_data)
    st.metric("Model Error (MSE)", f"{mse:.2f}")
