# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 09:30:00 2025

Enhanced Arcadis Cost Intelligence Suite with Mock Data and More Features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split # Added for better model evaluation
from sklearn.preprocessing import LabelEncoder # Added for handling categorical features if needed
import io
import base64
from datetime import datetime, timedelta
import time
import random
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import logging

# ------------------ ⚙️ APP CONFIGURATION ------------------
st.set_page_config(
    layout="wide",
    page_title="Arcadis Cost Intelligence Suite",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------ 🎨 STYLING & THEME ------------------
def apply_styling(dark_mode):
    """Applies CSS styling based on dark mode selection."""
    if dark_mode:
        st.markdown("""
            <style>
            .main { background-color: #263238; color: #FAFAFA; }
            .stTabs [data-baseweb="tab-list"] { gap: 2px; }
            .stTabs [data-baseweb="tab"] { background-color: #37474F; border-radius: 4px 4px 0 0; }
            .stTabs [aria-selected="true"] { background-color: #1E88E5; color: white; }
            .stMetric > div > div > div { color: #90CAF9; } /* Metric label color */
            .stMetric > label { color: #CFD8DC; } /* Metric value color */
            h1, h2, h3 { color: #1E88E5; }
            .stDataFrame { color: #FAFAFA; } /* Ensure dataframe text is visible */
             /* Style buttons */
            .stButton>button {
                border: 2px solid #1E88E5;
                border-radius: 5px;
                background-color: #1E88E5;
                color: white;
                padding: 8px 16px;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #1565C0;
                border-color: #1565C0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .stButton>button:active {
                background-color: #0D47A1;
                border-color: #0D47A1;
            }
            /* Style download links to look like buttons */
            a.download-button {
                display: inline-block;
                padding: 8px 16px;
                margin: 5px 0;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                background-color: #4CAF50;
                color: white;
                text-decoration: none;
                text-align: center;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            a.download-button:hover {
                background-color: #388E3C;
                border-color: #388E3C;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            </style>
        """, unsafe_allow_html=True)
    else:
         st.markdown("""
            <style>
            .stTabs [data-baseweb="tab-list"] { gap: 2px; }
            .stTabs [aria-selected="true"] { background-color: #E3F2FD; }
            h1, h2, h3 { color: #1976D2; }
             /* Style buttons */
            .stButton>button {
                border: 2px solid #1976D2;
                border-radius: 5px;
                background-color: #1976D2;
                color: white;
                padding: 8px 16px;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #1565C0;
                border-color: #1565C0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .stButton>button:active {
                background-color: #0D47A1;
                border-color: #0D47A1;
            }
            /* Style download links to look like buttons */
            a.download-button {
                display: inline-block;
                padding: 8px 16px;
                margin: 5px 0;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                background-color: #4CAF50;
                color: white;
                text-decoration: none;
                text-align: center;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            a.download-button:hover {
                background-color: #388E3C;
                border-color: #388E3C;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            </style>
        """, unsafe_allow_html=True)

# ------------------ 🔄 MOCK DATA GENERATION ------------------
@st.cache_data(ttl=3600) # Cache mock data for 1 hour
def generate_mock_data(num_rows=150):
    """Generates a realistic mock DataFrame for demonstration."""
    logging.info(f"Generating {num_rows} rows of mock data.")
    sectors = ['Infrastructure', 'Buildings', 'Water', 'Environment', 'Energy']
    asset_types = {
        'Infrastructure': ['Road', 'Rail', 'Airport', 'Bridge'],
        'Buildings': ['Commercial Office', 'Residential Tower', 'Hospital', 'School'],
        'Water': ['Treatment Plant', 'Pipeline Network', 'Reservoir'],
        'Environment': ['Remediation Site', 'Flood Defense', 'Waste Facility'],
        'Energy': ['Wind Farm', 'Solar Park', 'Substation']
    }
    locations = ['London', 'Manchester', 'Birmingham', 'Glasgow', 'Bristol', 'Leeds', 'Edinburgh']

    data = []
    start_date = datetime(2020, 1, 1)

    for i in range(num_rows):
        sector = random.choice(sectors)
        asset = random.choice(asset_types[sector])
        duration = random.randint(6, 48) # months
        cost = round(random.uniform(1.0, 150.0) + (duration / 12) * random.uniform(5, 20), 2) # Cost slightly correlated with duration
        manual_handling = round(random.uniform(5, 60), 1)
        quality_score = round(random.uniform(60, 98), 1)
        # Make overrun slightly correlated with complexity (cost/duration) and manual handling
        complexity_factor = (cost / duration) if duration > 0 else 0
        overrun_base = random.uniform(-5, 15)
        overrun = round(overrun_base + (manual_handling / 20) + (complexity_factor / 10) + random.gauss(0, 5), 1)
        proj_start_date = start_date + timedelta(days=random.randint(0, 1000))
        proj_end_date = proj_start_date + timedelta(days=duration * 30) # Approximate end date

        data.append({
            'Project ID': f'PROJ-{1001 + i}',
            'Sector': sector,
            'Asset Type': asset,
            'Location': random.choice(locations),
            'Cost (£m)': cost,
            'Duration (months)': duration,
            'Manual Handling %': manual_handling,
            'Quality Score': quality_score,
            'Forecasted Overrun (%)': overrun,
            'Start Date': proj_start_date.strftime('%Y-%m-%d'),
            'End Date': proj_end_date.strftime('%Y-%m-%d')
        })

    df = pd.DataFrame(data)
    # Ensure correct data types
    df['Cost (£m)'] = pd.to_numeric(df['Cost (£m)'])
    df['Duration (months)'] = pd.to_numeric(df['Duration (months)'])
    df['Manual Handling %'] = pd.to_numeric(df['Manual Handling %'])
    df['Quality Score'] = pd.to_numeric(df['Quality Score'])
    df['Forecasted Overrun (%)'] = pd.to_numeric(df['Forecasted Overrun (%)'])
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['End Date'] = pd.to_datetime(df['End Date'])
    logging.info("Mock data generation complete.")
    return df

# ------------------ 💾 DATA LOADING & VALIDATION ------------------
@st.cache_data(ttl=600) # Cache loaded data for 10 mins
def load_data(uploaded_file_content):
    """Loads data from uploaded Excel file."""
    logging.info("Loading data from uploaded file.")
    try:
        df = pd.read_excel(io.BytesIO(uploaded_file_content))
        logging.info(f"Successfully loaded {len(df)} rows from file.")
        # --- Basic Validation ---
        required_cols = ['Sector', 'Asset Type', 'Cost (£m)', 'Duration (months)']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Uploaded file is missing required columns: {', '.join(missing_cols)}. Please ensure your file has these columns.")
            return None
        # Attempt to convert key numeric columns, coercing errors
        for col in ['Cost (£m)', 'Duration (months)', 'Forecasted Overrun (%)', 'Manual Handling %', 'Quality Score']:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop rows where essential numeric columns couldn't be converted
        df.dropna(subset=['Cost (£m)', 'Duration (months)'], inplace=True)
        logging.info(f"Data loaded and basic validation passed. Rows after cleaning: {len(df)}")
        return df
    except Exception as e:
        st.error(f"Error loading or processing the Excel file: {e}")
        logging.error(f"File loading failed: {e}")
        return None

# ------------------ 🧪 DATA PROCESSING & ANALYSIS FUNCTIONS ------------------
@st.cache_data
def filter_data(data, sector_filter, asset_filter, cost_range, search_term):
    """Filters the DataFrame based on user selections."""
    # Start with a copy of the original data if it exists, otherwise an empty DataFrame
    if data is None:
        return pd.DataFrame()
    filtered = data.copy()

    # Apply filters safely checking if columns exist and filters are provided
    try:
        if 'Sector' in filtered.columns and sector_filter:
            filtered = filtered[filtered['Sector'].isin(sector_filter)]
        if 'Asset Type' in filtered.columns and asset_filter:
            filtered = filtered[filtered['Asset Type'].isin(asset_filter)]
        if 'Cost (£m)' in filtered.columns and cost_range:
             # Ensure cost_range is a tuple/list with two elements
             if isinstance(cost_range, (list, tuple)) and len(cost_range) == 2:
                 filtered = filtered[filtered['Cost (£m)'].between(cost_range[0], cost_range[1])]
             else:
                 logging.warning(f"Invalid cost_range provided: {cost_range}")

        if search_term:
            # Search across all columns converted to string
            filtered = filtered[filtered.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)]

    except Exception as e:
        st.warning(f"Error during filtering: {e}")
        logging.error(f"Filtering error: {e}")
        return data.copy() # Return unfiltered data on error to avoid crash

    logging.info(f"Data filtered. {len(filtered)} rows remaining.")
    return filtered

@st.cache_data
def forecast_cost(df):
    """Trains a RandomForest model to predict cost and calculates MSE."""
    required_features = ['Duration (months)', 'Manual Handling %', 'Quality Score']
    target = 'Cost (£m)'

    # Ensure df is a DataFrame before proceeding
    if not isinstance(df, pd.DataFrame):
        st.warning("Invalid data provided for forecasting.")
        return pd.DataFrame(), None, 0.0, pd.DataFrame()

    if not all(col in df.columns for col in required_features + [target]):
        st.warning(f"Required columns for forecasting ({', '.join(required_features + [target])}) not found. Skipping prediction.")
        # Ensure the original df is returned even if prediction is skipped
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(), None, 0.0, pd.DataFrame()

    df_clean = df.copy()
    # Ensure features are numeric, fill NaNs with median or mean
    for col in required_features + [target]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna(subset=required_features + [target])

    if len(df_clean) < 10: # Need sufficient data to train
        st.warning("Not enough data points (<10) after cleaning for cost forecasting.")
        return df.copy(), None, 0.0, pd.DataFrame()

    X = df_clean[required_features]
    y = df_clean[target]

    # Split data for a simple evaluation
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if len(X_train) < 2 or len(X_test) < 1:
             st.warning("Not enough data for train/test split.")
             return df.copy(), None, 0.0, pd.DataFrame()

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Predict on the entire clean dataset for display purposes
        predicted_all = model.predict(X)
        # Predict on test set for MSE calculation
        predicted_test = model.predict(X_test)
        mse = mean_squared_error(y_test, predicted_test)
        logging.info(f"Cost forecasting model trained. MSE on test set: {mse:.2f}")

        # Add predictions back to the original dataframe (align by index)
        predictions_series = pd.Series(predicted_all, index=X.index, name='Predicted Cost (£m)')
        # Use combine_first to handle potential index mismatches or NaNs introduced
        df_out = df.copy()
        df_out['Predicted Cost (£m)'] = predictions_series
        df_out.update(df_out[['Predicted Cost (£m)']].fillna(np.nan)) # Ensure NaNs remain where prediction wasn't possible

        # Get feature importances
        importances = pd.DataFrame({
            'Feature': required_features,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        return df_out, model, mse, importances
    except Exception as e:
        st.error(f"Error during model training or prediction: {e}")
        logging.error(f"Model training/prediction error: {e}")
        return df.copy(), None, 0.0, pd.DataFrame()


@st.cache_data
def detect_anomalies(df):
    """Uses Isolation Forest to detect anomalies based on Cost and Duration."""
    features = ['Cost (£m)', 'Duration (months)']

    # Ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        st.warning("Invalid data provided for anomaly detection.")
        return pd.DataFrame()

    # Add 'Anomaly' column immediately, default to False
    df_out = df.copy()
    if 'Anomaly' not in df_out.columns:
      df_out['Anomaly'] = False

    if not all(col in df_out.columns for col in features):
        st.warning(f"Required columns for anomaly detection ({', '.join(features)}) not found. Skipping.")
        return df_out # Return with 'Anomaly' column set to False

    df_clean = df_out.copy()
    # Ensure features are numeric, fill NaNs with median
    for col in features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)

    # Drop rows where features are still NaN after fill (shouldn't happen with median fill unless all values were NaN)
    df_clean = df_clean.dropna(subset=features)

    if len(df_clean) < 5: # Need some data for Isolation Forest
        st.warning("Not enough data points (<5) for anomaly detection.")
        return df_out # Return original df with 'Anomaly' as False

    X = df_clean[features]

    try:
        iso = IsolationForest(contamination='auto', random_state=42, n_estimators=100) # 'auto' contamination is often better
        anomaly_preds = iso.fit_predict(X)
        # Anomalies are marked as -1 by IsolationForest, normal as 1. Convert to boolean.
        anomaly_series = pd.Series(anomaly_preds == -1, index=X.index, name='Anomaly')

        # Update the 'Anomaly' column in df_out using the index alignment
        df_out['Anomaly'] = anomaly_series
        df_out['Anomaly'].fillna(False, inplace=True) # Ensure any rows not in df_clean are marked False

        logging.info(f"Anomaly detection complete. Found {df_out['Anomaly'].sum()} potential anomalies.")
    except Exception as e:
        st.error(f"Error during anomaly detection: {e}")
        logging.error(f"Anomaly detection error: {e}")
        # Ensure 'Anomaly' column exists and is False if error occurs
        df_out['Anomaly'] = False

    return df_out

def generate_ai_insight(df):
    """Generates a simple dynamic insight based on aggregated data."""
    if not isinstance(df, pd.DataFrame) or df.empty or 'Cost (£m)' not in df.columns or 'Forecasted Overrun (%)' not in df.columns:
        return "Arcadis AI Insight: Insufficient data for analysis."

    try:
        avg_cost = df['Cost (£m)'].mean()
        avg_overrun = df['Forecasted Overrun (%)'].mean()
        num_projects = len(df)

        insight = f"Arcadis AI Insight ({num_projects} projects): Avg cost £{avg_cost:,.2f}m, avg overrun {avg_overrun:.1f}%. "
        if avg_cost > 75 and avg_overrun > 10:
            insight += "High cost and overrun suggest focus on risk management for large projects."
        elif avg_cost > 50:
            insight += "Significant average cost suggests potential for value engineering and cost optimization strategies."
        elif avg_overrun > 15:
            insight += "High average overrun indicates a need to review project controls and scheduling."
        elif avg_overrun < 0:
            insight += "Projects are performing well under budget on average. Focus on maintaining efficiency."
        else:
            insight += "Performance appears stable. Consider exploring opportunities for further efficiency gains."
        return insight
    except Exception as e:
        logging.error(f"Error generating AI insight: {e}")
        return "Arcadis AI Insight: Error during analysis."


def generate_executive_summary(df):
    """Creates a brief executive summary text."""
    if not isinstance(df, pd.DataFrame) or df.empty or 'Cost (£m)' not in df.columns or 'Forecasted Overrun (%)' not in df.columns:
        return "Executive Summary: No data available to generate summary."

    try:
        total_value = df['Cost (£m)'].sum()
        avg_overrun = df['Forecasted Overrun (%)'].mean()
        num_projects = len(df)
        avg_duration_val = df['Duration (months)'].mean() if 'Duration (months)' in df.columns else None
        avg_duration = f"{avg_duration_val:.1f}" if avg_duration_val is not None else 'N/A'
        avg_cost_val = df['Cost (£m)'].mean()

        summary = f"Executive Summary:\n"
        summary += f"- Portfolio contains {num_projects} projects with a total value of £{total_value:,.2f}m.\n"
        summary += f"- Average project cost is £{avg_cost_val:,.2f}m.\n"
        if avg_duration != 'N/A':
            summary += f"- Average project duration is {avg_duration} months.\n"
        summary += f"- Average forecasted overrun is {avg_overrun:.1f}%.\n"

        recommendation = "Recommendations: "
        if avg_overrun > 10:
            recommendation += "Prioritize review of project controls and risk mitigation for projects with high overrun potential. "
        if avg_cost_val > 50:
            recommendation += "Investigate cost-saving opportunities and value engineering, especially for high-value assets. "
        if 'Quality Score' in df.columns and df['Quality Score'].mean() < 80:
            recommendation += "Address potential quality issues indicated by lower average scores. "
        if recommendation == "Recommendations: ":
            recommendation += "Maintain current performance levels and explore continuous improvement initiatives."

        return summary + recommendation
    except Exception as e:
        logging.error(f"Error generating executive summary: {e}")
        return "Executive Summary: Error during analysis."

def get_download_link(df, filename="arcadis_benchmarking_output.xlsx", link_text="📥 Download Excel"):
    """Generates a download link for a DataFrame as an Excel file."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No data to download."
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Convert datetime columns to string to avoid timezone issues in Excel
            df_copy = df.copy()
            for col in df_copy.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetimetz']).columns:
                try:
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                except AttributeError: # Handle cases where conversion might fail
                     logging.warning(f"Could not convert datetime column {col} to string for Excel export.")
                     df_copy[col] = df_copy[col].astype(str) # Fallback to string conversion

            df_copy.to_excel(writer, index=False, sheet_name='Data')
            # You could add more sheets here (e.g., summary stats)
        b64 = base64.b64encode(output.getvalue()).decode()
        # Use class="download-button" for styling
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-button">{link_text}</a>'
    except Exception as e:
        logging.error(f"Error generating Excel download link: {e}")
        return "Error creating download link."

def generate_pdf_report(df, filename="arcadis_benchmarking_report.pdf", link_text="📜 Download PDF Report"):
    """Generates a PDF report summarizing the data."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No data to generate report."
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Arcadis Cost Intelligence Report", styles['h1']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Summary Stats
        story.append(Paragraph("Portfolio Summary", styles['h2']))
        num_projects = len(df)
        total_value = df['Cost (£m)'].sum() if 'Cost (£m)' in df.columns else 0
        avg_cost = df['Cost (£m)'].mean() if 'Cost (£m)' in df.columns else 0
        avg_overrun = df['Forecasted Overrun (%)'].mean() if 'Forecasted Overrun (%)' in df.columns else 0
        avg_duration_val = df['Duration (months)'].mean() if 'Duration (months)' in df.columns else None
        avg_duration = f"{avg_duration_val:.1f} months" if avg_duration_val is not None else "N/A"

        summary_data = [
            ['Metric', 'Value'],
            ['Number of Projects', f"{num_projects:,}"],
            ['Total Portfolio Value', f"£{total_value:,.2f}m"],
            ['Average Project Cost', f"£{avg_cost:,.2f}m"],
            ['Average Duration', avg_duration],
            ['Average Forecasted Overrun', f"{avg_overrun:.1f}%" if not pd.isna(avg_overrun) else "N/A"],
        ]
        summary_table = Table(summary_data, colWidths=[200, 150])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 12))

        # Executive Summary Text
        story.append(Paragraph("Executive Summary", styles['h2']))
        exec_summary_text = generate_executive_summary(df).replace('\n', '<br/>') # Use <br/> for line breaks in PDF
        story.append(Paragraph(exec_summary_text, styles['Normal']))
        story.append(Spacer(1, 24))

        # Add a snippet of the data (first 10 rows)
        story.append(Paragraph("Data Sample (First 10 Projects)", styles['h3']))
        df_sample = df.head(10).copy()
        # Select and rename columns for the table
        cols_to_show = ['Project ID', 'Sector', 'Asset Type', 'Cost (£m)', 'Duration (months)', 'Forecasted Overrun (%)']
        # Filter df_sample to only include columns that actually exist
        df_sample = df_sample[[col for col in cols_to_show if col in df_sample.columns]]

        # Convert data to list of lists for ReportLab table, including header
        data_list = [df_sample.columns.tolist()] + df_sample.astype(str).values.tolist()

        if len(data_list) > 1: # Check if there's data besides the header
            data_table = Table(data_list, hAlign='LEFT')
            data_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 8) # Smaller font for data
            ]))
            story.append(data_table)
        else:
            story.append(Paragraph("No data available for sample.", styles['Normal']))


        doc.build(story)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        # Use class="download-button" for styling
        return f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-button">{link_text}</a>'
    except Exception as e:
        logging.error(f"Error generating PDF report: {e}")
        st.error(f"Could not generate PDF report: {e}")
        return "Error creating PDF report link."

# ------------------ ✨ APP LAYOUT & UI ELEMENTS ------------------

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/10/Arcadis_Logo.svg", width=150)
    st.markdown("<h2 style='color: #1E88E5;'>Arcadis Control Panel</h2>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚙️ Settings")
    dark_mode = st.checkbox("🌙 Dark Mode", value=True) # Default to dark mode

    st.markdown("---")
    st.subheader("💾 Data Input")
    uploaded_file = st.file_uploader("📤 Upload Your Dataset (Excel)", type=["xlsx", "xls"])
    # Checkbox default depends only on whether a file has been uploaded in this session run
    use_mock_data_default = (uploaded_file is None)
    use_mock_data = st.checkbox("🔄 Use Mock Data Instead", value=use_mock_data_default)

    # --- Refined Data Loading Logic ---
    raw_data = None
    data_source = ""

    if use_mock_data:
        st.info("Using built-in mock data.")
        raw_data = generate_mock_data()
        data_source = "Mock Data"
    elif uploaded_file is not None:
        st.info(f"Processing uploaded file: {uploaded_file.name}")
        raw_data = load_data(uploaded_file.getvalue())
        if raw_data is not None:
            data_source = f"Uploaded File: {uploaded_file.name}"
        else:
            # load_data() should show an error, but we can add a fallback message here
            st.error("Failed to load the uploaded file. Please check the file format and required columns.")
            data_source = "File Load Error"
            # Optionally fall back to mock data if upload fails
            # raw_data = generate_mock_data()
            # data_source = "Mock Data (fallback)"
    else:
        # This state should ideally only be reached if the user unchecks mock data without uploading a file
        st.warning("Please upload an Excel file or check 'Use Mock Data Instead'.")
        data_source = "No Data Source Selected"

    st.markdown(f"**Current Data Source:** {data_source}")

    # --- Filtering Section (Conditional on raw_data) ---
    # Initialize filter variables outside the 'if' block
    sector_filter = []
    asset_filter = []
    cost_range = None
    search_term = ""
    filtered_data = pd.DataFrame() # Default to empty DataFrame

    if raw_data is not None and not raw_data.empty:
        st.markdown("---")
        st.subheader("🔍 Filter Data")

        # Define filter widgets using data from raw_data
        # Sector Filter
        if 'Sector' in raw_data.columns:
            available_sectors = sorted(raw_data['Sector'].unique())
            default_sectors = available_sectors # Select all by default
            sector_filter = st.multiselect("Sectors", available_sectors, default=default_sectors)
        else:
            st.warning("Column 'Sector' not found in data.")

        # Asset Type Filter (Dynamic based on selected sectors)
        if 'Asset Type' in raw_data.columns:
            if 'Sector' in raw_data.columns and sector_filter:
                 # Filter available assets based on selected sectors first
                 relevant_assets = raw_data[raw_data['Sector'].isin(sector_filter)]['Asset Type'].unique()
                 available_assets = sorted(relevant_assets)
            else:
                 # If no sector filter or no sector column, show all assets
                 available_assets = sorted(raw_data['Asset Type'].unique())

            default_assets = available_assets # Select all available by default
            asset_filter = st.multiselect("Asset Types", available_assets, default=default_assets)
        else:
            st.warning("Column 'Asset Type' not found.")

        # Cost Range Filter
        if 'Cost (£m)' in raw_data.columns:
            min_cost = float(raw_data['Cost (£m)'].min())
            max_cost = float(raw_data['Cost (£m)'].max())
            # Ensure min/max are valid before creating slider
            if min_cost <= max_cost:
                cost_range = st.slider("Cost Range (£m)", min_value=min_cost, max_value=max_cost, value=(min_cost, max_cost))
            else:
                 st.warning("Invalid cost range in data (min > max). Cannot create slider.")
                 cost_range = (min_cost, max_cost) # Provide a default tuple even if invalid
        else:
            st.warning("Column 'Cost (£m)' not found.")

        # Search Term
        search_term = st.text_input("Search Projects (any field)")

        # Apply filters AFTER widgets are defined
        filtered_data = filter_data(raw_data, sector_filter, asset_filter, cost_range, search_term)

    else:
        # If raw_data is None or empty, ensure filtered_data is empty
        st.warning("No raw data loaded. Filtering is disabled.")
        filtered_data = pd.DataFrame()


    # --- AI Assistant Placeholder ---
    st.markdown("---")
    st.subheader("🤖 Arcadis AI Assistant")
    question = st.text_input("Ask about your data (e.g., 'show high risk projects')")
    if st.button("Ask AI"):
        # Simulate AI response (replace with actual AI call if available)
        if question:
             st.info(f"AI Response: Analyzing '{question}'... Feature coming soon!")
             # Example placeholder logic (ensure filtered_data is checked)
             if not filtered_data.empty:
                 if "high risk" in question.lower() and 'Forecasted Overrun (%)' in filtered_data.columns:
                     high_risk = filtered_data[filtered_data['Forecasted Overrun (%)'] > 15]
                     st.write(f"Found {len(high_risk)} projects with >15% overrun potential:")
                     st.dataframe(high_risk[['Project ID', 'Sector', 'Cost (£m)', 'Forecasted Overrun (%)']].head())
                 elif "costliest" in question.lower() and 'Cost (£m)' in filtered_data.columns:
                     costliest = filtered_data.nlargest(5, 'Cost (£m)')
                     st.write("Top 5 costliest projects:")
                     st.dataframe(costliest[['Project ID', 'Sector', 'Cost (£m)']])
                 else:
                     st.write("AI analysis simulation complete. More detailed insights require full integration.")
             else:
                 st.warning("Cannot perform AI analysis as no data is currently loaded or filtered.")
        else:
            st.warning("Please enter a question for the AI assistant.")


# Apply styling based on sidebar choice
apply_styling(dark_mode)

# --- Main Panel ---
st.title("🚀 Arcadis Cost Intelligence Suite")

# --- Loading Animation ---
# Use session state to ensure this runs only once per session
if "app_initialized" not in st.session_state:
    with st.spinner("🔄 Initializing Dashboard... Please wait."):
        loading_msg = st.empty()
        progress_bar = st.progress(0)
        for percent in range(101):
            time.sleep(0.01) # Faster load simulation
            progress_bar.progress(percent)
            # loading_msg.text(f"Loading components... {percent}%") # Commented out for less flicker
        time.sleep(0.2)
        loading_msg.empty()
        progress_bar.empty()
    st.session_state.app_initialized = True # Mark as initialized


# --- Check if data exists before creating tabs ---
# Check the filtered_data DataFrame which is now reliably initialized
if filtered_data.empty:
    st.error("No data available to display. Please check your filters, upload a valid file, or ensure 'Use Mock Data' is selected.")
    # Optionally display the landing page graphic again or stop
    st.markdown("""
        <div style='text-align: center; padding: 30px;'>
            <h2 style='color:#FF7043;'>No Data Loaded or Filtered</h2>
            <p style='font-size:16px; color: #757575;'>Upload data or use mock data via the sidebar to begin analysis. Check filters if data source is selected.</p>
            <img src='https://media.giphy.com/media/3o7TKSjRrfIPjeiVyE/giphy.gif' width='250'>
        </div>
    """, unsafe_allow_html=True)
    st.stop() # Stop execution if no data

# --- Create Tabs ---
tab_titles = ["🌟 Overview", "📈 Executive Insights", "🔍 Deep Dive", "🔮 Predictive Analytics"]
tabs = st.tabs(tab_titles)

# --- Tab 1: Overview ---
with tabs[0]:
    st.header("🌟 Portfolio Snapshot")
    st.markdown("Key metrics and a summary of the currently filtered dataset.")

    # AI Insight Banner
    ai_insight = generate_ai_insight(filtered_data)
    st.info(ai_insight) # Use st.info for better visibility

    # Key Metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    num_projects = len(filtered_data)
    # Add checks for column existence before calculation
    total_value = filtered_data['Cost (£m)'].sum() if 'Cost (£m)' in filtered_data.columns else 0
    avg_overrun_val = filtered_data['Forecasted Overrun (%)'].mean() if 'Forecasted Overrun (%)' in filtered_data.columns else None
    avg_duration_val = filtered_data['Duration (months)'].mean() if 'Duration (months)' in filtered_data.columns else None

    avg_overrun_display = f"{avg_overrun_val:.1f}%" if avg_overrun_val is not None and not pd.isna(avg_overrun_val) else "N/A"
    avg_duration_display = f"{avg_duration_val:.1f} mo" if avg_duration_val is not None and not pd.isna(avg_duration_val) else "N/A"
    delta_display = f"{avg_overrun_val-5:.1f}% vs Target (5%)" if avg_overrun_val is not None and not pd.isna(avg_overrun_val) else None


    col1.metric("Total Projects", f"{num_projects:,}")
    col2.metric("Total Value", f"£{total_value:,.2f}m")
    col3.metric("Avg Duration", avg_duration_display)
    col4.metric("Avg Overrun", avg_overrun_display, delta=delta_display, delta_color="inverse")

    st.markdown("---")

    # Download Links
    st.subheader("⬇️ Downloads")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.markdown(get_download_link(filtered_data, filename="arcadis_filtered_data.xlsx"), unsafe_allow_html=True)
    with col_dl2:
        st.markdown(generate_pdf_report(filtered_data, filename="arcadis_summary_report.pdf"), unsafe_allow_html=True)

    st.markdown("---")

    # Data Preview Table
    st.subheader("📋 Data Preview")
    st.dataframe(filtered_data.head(10)) # Show first 10 rows
    st.caption(f"Displaying first 10 of {len(filtered_data)} filtered projects.")


# --- Tab 2: Executive Insights ---
with tabs[1]:
    st.header("📈 Executive Insights & Summaries")
    st.markdown("High-level analysis and textual summaries for quick understanding.")

    st.subheader("📝 Summary")
    summary_text = generate_executive_summary(filtered_data)
    st.text_area("Executive Summary:", value=summary_text, height=200, disabled=True, key="exec_summary_textarea") # Use text_area for better display

    st.markdown("---")
    st.subheader("📊 Cost Distribution by Sector")
    if 'Sector' in filtered_data.columns and 'Cost (£m)' in filtered_data.columns:
        # Ensure there's data to plot after filtering
        if not filtered_data.empty:
            try:
                # Use Plotly for interactive charts
                fig_sector_cost = px.pie(filtered_data, names='Sector', values='Cost (£m)',
                                         title='Total Project Value (£m) by Sector',
                                         hole=0.3) # Donut chart
                fig_sector_cost.update_traces(textposition='inside', textinfo='percent+label')
                fig_sector_cost.update_layout(showlegend=False) # Legend can be redundant with labels
                st.plotly_chart(fig_sector_cost, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate Sector Cost chart: {e}")
                logging.error(f"Sector Cost chart error: {e}")
        else:
            st.info("No data matching current filters to display Sector Cost chart.")
    else:
        st.warning("Required columns ('Sector', 'Cost (£m)') not available for Sector Cost chart.")

    st.markdown("---")
    st.subheader("⏱️ Average Duration by Asset Type")
    if 'Asset Type' in filtered_data.columns and 'Duration (months)' in filtered_data.columns:
         if not filtered_data.empty:
             try:
                avg_duration_asset = filtered_data.groupby('Asset Type')['Duration (months)'].mean().reset_index().sort_values(by='Duration (months)', ascending=False)
                fig_asset_dur = px.bar(avg_duration_asset.head(15), # Show top 15 longest avg duration
                                       x='Asset Type', y='Duration (months)',
                                       title='Average Project Duration (Months) by Asset Type (Top 15)',
                                       labels={'Duration (months)': 'Avg Duration (Months)', 'Asset Type': 'Asset Type'},
                                       color='Duration (months)', color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig_asset_dur, use_container_width=True)
             except Exception as e:
                st.error(f"Could not generate Asset Duration chart: {e}")
                logging.error(f"Asset Duration chart error: {e}")
         else:
            st.info("No data matching current filters to display Asset Duration chart.")
    else:
        st.warning("Required columns ('Asset Type', 'Duration (months)') not available for Asset Duration chart.")


# --- Tab 3: Deep Dive ---
with tabs[2]:
    st.header("🔍 Deep Dive Analytics")
    st.markdown("Explore relationships and distributions within your data.")

    if filtered_data.empty:
        st.info("No data matching current filters to display deep dive analytics.")
    else:
        col_dd1, col_dd2 = st.columns(2)

        with col_dd1:
            st.subheader("Cost vs Duration Scatter")
            if 'Cost (£m)' in filtered_data.columns and 'Duration (months)' in filtered_data.columns:
                try:
                    # Enhanced Scatter Plot
                    hover_cols = ['Project ID']
                    if 'Asset Type' in filtered_data.columns: hover_cols.append('Asset Type')
                    if 'Forecasted Overrun (%)' in filtered_data.columns: hover_cols.append('Forecasted Overrun (%)')

                    fig_scatter = px.scatter(filtered_data,
                                             x='Duration (months)',
                                             y='Cost (£m)',
                                             color='Sector' if 'Sector' in filtered_data.columns else None,
                                             size='Cost (£m)' if 'Cost (£m)' in filtered_data.columns else None, # Size bubbles by cost
                                             hover_data=hover_cols,
                                             title="Project Cost vs. Duration by Sector")
                    fig_scatter.update_layout(xaxis_title="Duration (Months)", yaxis_title="Cost (£ Million)")
                    st.plotly_chart(fig_scatter, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate Cost vs Duration scatter plot: {e}")
                    logging.error(f"Cost vs Duration scatter error: {e}")
            else:
                 st.warning("Required columns ('Cost (£m)', 'Duration (months)') not available for scatter plot.")

            st.subheader("Cost Distribution")
            if 'Cost (£m)' in filtered_data.columns:
                try:
                    fig_hist = px.histogram(filtered_data, x='Cost (£m)', nbins=30, title="Distribution of Project Costs")
                    fig_hist.update_layout(xaxis_title="Cost (£ Million)", yaxis_title="Number of Projects")
                    st.plotly_chart(fig_hist, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate Cost Distribution histogram: {e}")
                    logging.error(f"Cost Distribution histogram error: {e}")
            else:
                st.warning("Column 'Cost (£m)' not available for histogram.")


        with col_dd2:
            st.subheader("Duration by Sector")
            if 'Duration (months)' in filtered_data.columns and 'Sector' in filtered_data.columns:
                try:
                    fig_box = px.box(filtered_data, x='Sector', y='Duration (months)',
                                     color='Sector', title="Project Duration Distribution by Sector")
                    fig_box.update_layout(xaxis_title="Sector", yaxis_title="Duration (Months)")
                    st.plotly_chart(fig_box, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate Duration by Sector box plot: {e}")
                    logging.error(f"Duration by Sector box plot error: {e}")
            else:
                st.warning("Required columns ('Duration (months)', 'Sector') not available for box plot.")

            st.subheader("Correlation Heatmap")
            numeric_cols = filtered_data.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) > 1:
                try:
                    corr = filtered_data[numeric_cols].corr()
                    fig_heatmap = go.Figure(data=go.Heatmap(
                                       z=corr.values,
                                       x=corr.columns,
                                       y=corr.columns,
                                       colorscale='Blues', # Choose a colorscale
                                       colorbar=dict(title='Correlation'),
                                       zmin=-1, zmax=1)) # Set scale from -1 to 1
                    fig_heatmap.update_layout(title='Correlation Matrix of Numeric Features')
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate Correlation Heatmap: {e}")
                    logging.error(f"Correlation Heatmap error: {e}")
            else:
                st.warning("Not enough numeric columns found for correlation analysis.")


# --- Tab 4: Predictive Analytics ---
with tabs[3]:
    st.header("🔮 Predictive Analytics")
    st.markdown("Forecast project costs and identify potential anomalies using machine learning models.")

    if filtered_data.empty:
         st.info("No data matching current filters to display predictive analytics.")
    else:
        # --- Cost Forecasting ---
        st.subheader("💰 Cost Forecasting (Random Forest)")
        # Run forecasting (function handles caching and checks)
        forecasted_data, model, mse, importances = forecast_cost(filtered_data)

        if model is not None: # Check if model was trained successfully
            st.metric("Model Mean Squared Error (MSE)", f"{mse:.2f}")
            st.caption("MSE indicates the average squared difference between predicted and actual costs on a test set. Lower is better.")

            # Display Feature Importances
            st.subheader("Feature Importance for Cost Prediction")
            if not importances.empty:
                 try:
                     fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', title='Feature Importance')
                     st.plotly_chart(fig_imp, use_container_width=True)
                 except Exception as e:
                     st.error(f"Could not display Feature Importance chart: {e}")
                     logging.error(f"Feature Importance chart error: {e}")
            else:
                 st.info("Feature importances could not be calculated.")

            # Display Forecasted Data
            st.subheader("Data with Predicted Costs")
            cols_to_show = ['Project ID', 'Cost (£m)', 'Predicted Cost (£m)', 'Duration (months)', 'Manual Handling %', 'Quality Score']
            display_cols = [col for col in cols_to_show if col in forecasted_data.columns]
            if display_cols:
                st.dataframe(forecasted_data[display_cols].head(10))
                st.caption("Showing first 10 rows with actual and predicted costs (where available).")
            else:
                st.warning("Could not display predicted costs table due to missing columns.")

        else:
            # Warning is already shown inside forecast_cost function if it fails
            st.info("Cost forecasting model could not be trained with the current data/filters.")

        st.markdown("---")

        # --- Anomaly Detection ---
        st.subheader("❗ Anomaly Detection (Isolation Forest)")
        st.markdown("Identifies projects that are unusual based on their Cost and Duration.")
        # Run anomaly detection (function handles caching and checks)
        # Use forecasted_data which might have 'Predicted Cost (£m)' but detection only uses Cost and Duration
        data_with_anomalies = detect_anomalies(forecasted_data)

        if 'Anomaly' in data_with_anomalies.columns:
            num_anomalies = data_with_anomalies['Anomaly'].sum()
            st.metric("Potential Anomalies Detected", f"{num_anomalies}")

            # Plot anomalies
            st.subheader("Anomaly Visualization (Cost vs Duration)")
            if 'Cost (£m)' in data_with_anomalies.columns and 'Duration (months)' in data_with_anomalies.columns:
                 try:
                    # Create a color mapping for anomalies
                    # Ensure 'Anomaly Label' exists even if no anomalies are found
                    data_with_anomalies['Anomaly Label'] = data_with_anomalies['Anomaly'].map({True: 'Anomaly', False: 'Normal'}).fillna('Normal')

                    fig_anomaly = px.scatter(data_with_anomalies,
                                             x='Duration (months)',
                                             y='Cost (£m)',
                                             color='Anomaly Label',
                                             color_discrete_map={'Anomaly': 'red', 'Normal': 'blue'},
                                             title='Anomaly Detection: Cost vs Duration',
                                             hover_data=['Project ID', 'Sector'] if all(c in data_with_anomalies.columns for c in ['Project ID', 'Sector']) else ['Project ID'],
                                             symbol='Anomaly Label', # Use different symbols
                                             symbol_map={'Anomaly': 'x', 'Normal': 'circle'})
                    fig_anomaly.update_layout(xaxis_title="Duration (Months)", yaxis_title="Cost (£ Million)")
                    st.plotly_chart(fig_anomaly, use_container_width=True)
                 except Exception as e:
                     st.error(f"Could not generate Anomaly Visualization plot: {e}")
                     logging.error(f"Anomaly Visualization plot error: {e}")

                 # Display anomalous projects
                 st.subheader("Anomalous Projects Details")
                 anomalous_projects = data_with_anomalies[data_with_anomalies['Anomaly'] == True]
                 if not anomalous_projects.empty:
                     cols_to_show_anomaly = ['Project ID', 'Sector', 'Asset Type', 'Cost (£m)', 'Duration (months)', 'Forecasted Overrun (%)']
                     display_anomaly_cols = [col for col in cols_to_show_anomaly if col in anomalous_projects.columns]
                     if display_anomaly_cols:
                        st.dataframe(anomalous_projects[display_anomaly_cols])
                     else:
                        st.warning("Could not display anomaly details table due to missing columns.")
                 else:
                     st.info("No anomalies detected in the current filtered data.")
            else:
                 st.warning("Required columns ('Cost (£m)', 'Duration (months)') not available for anomaly visualization.")

        else:
            # Warning is shown inside detect_anomalies if it fails
             st.info("Anomaly detection could not be performed.")


# --- Footer ---
st.markdown("---")
st.caption("Arcadis Cost Intelligence Suite © 2025 - Empowering Data-Driven Decisions")
